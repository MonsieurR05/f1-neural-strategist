"""Download races into per-race raw Parquet files, using the source configured for each season.

Usage (from the repository root):

    python -m strategist.data.ingest --config configs/data.yaml
    python -m strategist.data.ingest --config configs/data.yaml --seasons 2024 --event Bahrain

Output, one directory per race at ``<raw>/<year>/<round:02d>/``:

    laps.parquet     laps joined with per-lap weather (schema: strategist.data.sources)
    results.parquet  classification, grid, status
    quali.parquet    best qualifying lap per driver (the pre-race pace prior)
    meta.json        event info and data source; written last, so it marks a complete race

The run is idempotent: complete races are skipped, so an interrupted run can be restarted.
"""

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import fastf1
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

from strategist.config import load_config
from strategist.data.sources import RaceData, Source, TimingUnavailableError, make_source

log = logging.getLogger(__name__)

RATE_LIMIT_WAIT_S = 61 * 60
RATE_LIMIT_RETRIES = 3


def race_dir(raw_root: Path, year: int, round_number: int) -> Path:
    return raw_root / str(year) / f"{round_number:02d}"


def season_rounds(year: int, event_filter: str | None = None) -> list[pd.Series]:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    events = [row for _, row in schedule.iterrows() if row["RoundNumber"] > 0]
    if event_filter:
        needle = event_filter.lower()
        events = [
            e for e in events
            if any(needle in str(e[c]).lower() for c in ("EventName", "Location", "Country"))
        ]
    return events


def fetch_respecting_limits(source: Source, year: int, event: pd.Series, wait: bool) -> RaceData:
    """FastF1 counts its own calls (500/h overall, 200/h to Jolpica) and raises when over.
    With ``wait``, sleep until the hourly window has cleared and try the race again."""
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            return source.fetch(year, event)
        except RateLimitExceededError as exc:
            if not wait or attempt == RATE_LIMIT_RETRIES:
                raise
            log.warning("rate limit hit (%s); waiting %d min", exc, RATE_LIMIT_WAIT_S // 60)
            time.sleep(RATE_LIMIT_WAIT_S)
    raise AssertionError("unreachable")


def write_race(race: RaceData, year: int, event: pd.Series, source: str, raw_root: Path) -> None:
    rnd = int(event["RoundNumber"])
    out = race_dir(raw_root, year, rnd)
    out.mkdir(parents=True, exist_ok=True)

    laps = race.laps.copy()
    laps.insert(0, "Year", year)
    laps.insert(1, "Round", rnd)
    laps.to_parquet(out / "laps.parquet", index=False)
    race.results.to_parquet(out / "results.parquet", index=False)
    race.quali.to_parquet(out / "quali.parquet", index=False)

    meta = {
        **race.meta,
        "year": year,
        "round": rnd,
        "event_name": event["EventName"],
        "location": event["Location"],
        "country": event["Country"],
        "event_date": event["EventDate"].date().isoformat(),
        "source": source,
        "fastf1_version": fastf1.__version__,
        "ingested_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--seasons", type=int, nargs="*", help="override config seasons")
    parser.add_argument("--event", help="substring match on event name, location or country")
    parser.add_argument("--wait-on-rate-limit", action="store_true",
                        help="when FastF1's hourly API limit is hit, wait for it to clear and resume")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fastf1.set_log_level("WARNING")

    cfg = load_config(args.config)
    cache = Path(cfg["paths"]["fastf1_cache"])
    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.configure(cache_dir=str(cache))
    raw_root = Path(cfg["paths"]["raw"])

    source_by_year = {y: name for name, years in cfg["sources"].items() for y in years}
    sources = {}
    failures = []
    for year in args.seasons or sorted(source_by_year):
        if year not in source_by_year:
            log.error("no data source configured for %d", year)
            return 1
        name = source_by_year[year]
        source = sources.setdefault(name, make_source(name, cfg))
        for event in season_rounds(year, args.event):
            rnd = int(event["RoundNumber"])
            label = f"{year} R{rnd:02d} {event['EventName']}"
            if (race_dir(raw_root, year, rnd) / "meta.json").exists():
                log.info("skip  %s (already ingested)", label)
                continue
            log.info("fetch %s [%s]", label, name)
            try:
                write_race(fetch_respecting_limits(source, year, event, args.wait_on_rate_limit),
                           year, event, name, raw_root)
            except RateLimitExceededError as exc:
                log.error("API rate limit hit (%s). Re-run later to resume.", exc)
                return 2
            except TimingUnavailableError as exc:
                log.error("failed %s: %s", label, exc)
                failures.append({"year": year, "round": rnd, "event": label, "error": str(exc)})
            except Exception as exc:
                log.exception("failed %s", label)
                failures.append({"year": year, "round": rnd, "event": label, "error": repr(exc)})

    failures_path = raw_root / "failures.json"
    if failures:
        failures_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        log.warning("%d race(s) failed; see %s", len(failures), failures_path)
        return 1
    failures_path.unlink(missing_ok=True)
    log.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
