"""Download race sessions from FastF1 into per-race raw Parquet files.

Usage (from the repository root):

    python -m strategist.data.ingest --config configs/data.yaml
    python -m strategist.data.ingest --config configs/data.yaml --seasons 2024 --event Bahrain

Output, one directory per race at ``<raw>/<year>/<round:02d>/``:

    laps.parquet     FastF1 laps joined with per-lap weather
    results.parquet  classification, grid, status
    quali.parquet    best qualifying lap per driver (the pre-race pace prior)
    meta.json        event info; written last, so its presence marks a complete race

Timedelta columns are stored as float seconds; column names are otherwise FastF1's.
The run is idempotent: complete races are skipped, so an interrupted run can be restarted.
"""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import fastf1
import pandas as pd
from fastf1.exceptions import DataNotLoadedError, RateLimitExceededError

from strategist.config import load_config

log = logging.getLogger(__name__)


class TimingUnavailableError(RuntimeError):
    """FastF1 could not get lap timing for a session (not a transient error)."""

WEATHER_COLUMNS = ["AirTemp", "TrackTemp", "Humidity", "Pressure", "Rainfall", "WindSpeed"]
RESULT_COLUMNS = [
    "DriverNumber", "Abbreviation", "TeamName", "Position", "ClassifiedPosition",
    "GridPosition", "Status", "Points", "Laps", "Time",
]


def timedeltas_to_seconds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="timedelta64[ns]").columns:
        df[col] = df[col].dt.total_seconds()
    return df


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


def fetch_race_laps(year: int, round_number: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    session = fastf1.get_session(year, round_number, "R")
    session.load(laps=True, telemetry=False, weather=True, messages=False)

    try:
        laps = pd.DataFrame(session.laps).reset_index(drop=True)
    except DataNotLoadedError:
        raise TimingUnavailableError(
            "no timing data: livetiming.formula1.com needs an F1 TV sign-in "
            "(F1_USERNAME/F1_PASSWORD) and the FastF1 mirror lacks this session"
        ) from None
    try:
        weather = session.laps.get_weather_data().reset_index(drop=True)
        laps = pd.concat([laps, weather.reindex(columns=WEATHER_COLUMNS)], axis=1)
    except Exception as exc:  # weather feed is missing for a few sessions
        log.warning("%s R%02d: no weather data (%s)", year, round_number, exc)
        for col in WEATHER_COLUMNS:
            laps[col] = pd.NA
    laps = timedeltas_to_seconds(laps)

    results = timedeltas_to_seconds(
        pd.DataFrame(session.results).reindex(columns=RESULT_COLUMNS).reset_index(drop=True)
    )

    meta = {
        "total_laps": session.total_laps,
        "laps_completed": int(laps["LapNumber"].max()) if len(laps) else 0,
    }
    return laps, results, meta


def fetch_quali_best(year: int, round_number: int) -> pd.DataFrame:
    session = fastf1.get_session(year, round_number, "Q")
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    laps = session.laps
    valid = laps[laps["Deleted"].ne(True) & laps["LapTime"].notna()]
    best = valid.groupby("Driver")["LapTime"].min().dt.total_seconds()
    return best.rename("QualiBestS").reset_index()


def ingest_race(year: int, event: pd.Series, raw_root: Path) -> None:
    rnd = int(event["RoundNumber"])
    laps, results, meta = fetch_race_laps(year, rnd)
    out = race_dir(raw_root, year, rnd)
    out.mkdir(parents=True, exist_ok=True)
    laps.insert(0, "Year", year)
    laps.insert(1, "Round", rnd)
    laps.to_parquet(out / "laps.parquet", index=False)
    results.to_parquet(out / "results.parquet", index=False)

    try:
        quali = fetch_quali_best(year, rnd)
    except RateLimitExceededError:
        raise
    except Exception as exc:
        log.warning("%s R%02d: qualifying unavailable (%s)", year, rnd, exc)
        quali = pd.DataFrame({"Driver": pd.Series(dtype=str), "QualiBestS": pd.Series(dtype=float)})
    quali.to_parquet(out / "quali.parquet", index=False)

    meta.update({
        "year": year,
        "round": rnd,
        "event_name": event["EventName"],
        "location": event["Location"],
        "country": event["Country"],
        "event_date": event["EventDate"].date().isoformat(),
        "fastf1_version": fastf1.__version__,
        "ingested_at": datetime.now(UTC).isoformat(timespec="seconds"),
    })
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--seasons", type=int, nargs="*", help="override config seasons")
    parser.add_argument("--event", help="substring match on event name, location or country")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fastf1.set_log_level("WARNING")

    cfg = load_config(args.config)
    cache = Path(cfg["paths"]["fastf1_cache"])
    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.configure(cache_dir=str(cache))
    raw_root = Path(cfg["paths"]["raw"])

    failures = []
    for year in args.seasons or cfg["seasons"]:
        for event in season_rounds(year, args.event):
            rnd = int(event["RoundNumber"])
            label = f"{year} R{rnd:02d} {event['EventName']}"
            if (race_dir(raw_root, year, rnd) / "meta.json").exists():
                log.info("skip  %s (already ingested)", label)
                continue
            log.info("fetch %s", label)
            try:
                ingest_race(year, event, raw_root)
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
