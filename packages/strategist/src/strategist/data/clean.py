"""Turn raw per-race Parquet into one model-ready lap table.

Usage (from the repository root):

    python -m strategist.data.clean --config configs/data.yaml

Writes ``<processed>/laps.parquet`` (every lap, with features and an ``exclude_reason``)
and ``<processed>/races.parquet`` (one row per race). Nothing is dropped: the simulator
needs pit and safety-car laps, so training code selects ``exclude_reason.isna()``.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from strategist.config import load_config

log = logging.getLogger(__name__)

SLICKS = {"SOFT", "MEDIUM", "HARD"}
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}

# Checked in this order; a lap gets the first reason that applies.
EXCLUSION_ORDER = [
    "wet_race", "no_time", "generated", "lap1", "pit_in", "pit_out",
    "neutralised", "non_slick", "inaccurate", "slow_outlier",
]

RENAME = {
    "Year": "year", "Round": "round", "Driver": "driver", "DriverNumber": "driver_number",
    "Team": "team", "LapNumber": "lap_number", "LapTime": "lap_time_s", "Time": "session_time_s",
    "Stint": "stint", "Compound": "compound", "TyreLife": "tyre_life", "FreshTyre": "fresh_tyre",
    "Position": "position", "TrackStatus": "track_status", "PitInTime": "pit_in_time_s",
    "PitOutTime": "pit_out_time_s", "Sector1Time": "sector1_s", "Sector2Time": "sector2_s",
    "Sector3Time": "sector3_s", "IsAccurate": "is_accurate", "FastF1Generated": "generated",
    "AirTemp": "air_temp", "TrackTemp": "track_temp", "Humidity": "humidity",
    "Pressure": "pressure", "Rainfall": "rainfall", "WindSpeed": "wind_speed",
}


def circuit_key(meta: dict, aliases: dict[str, str]) -> str:
    """Stable circuit identifier: event-name aliases first (layout variants), then location."""
    return aliases.get(meta["event_name"]) or aliases.get(meta["location"]) or meta["location"]


def add_race_features(laps: pd.DataFrame, total_laps: int) -> pd.DataFrame:
    laps = laps.copy()
    laps["total_laps"] = total_laps
    laps["laps_remaining"] = total_laps - laps["lap_number"]
    laps["fuel_frac"] = (laps["laps_remaining"] / total_laps).clip(lower=0)
    return add_intervals(laps)


def add_intervals(laps: pd.DataFrame) -> pd.DataFrame:
    """Gap to the car directly ahead, measured at the line.

    ``interval_s`` is at the end of the lap (it depends on the lap itself);
    ``interval_start_s`` is the previous lap's value, i.e. known when the lap starts, and is the
    one models may use. Leaders get NaN (free air).
    """
    laps = laps.sort_values(["lap_number", "session_time_s"]).copy()
    laps["interval_s"] = laps.groupby("lap_number")["session_time_s"].diff()
    laps = laps.sort_values(["driver", "lap_number"])
    laps["interval_start_s"] = laps.groupby("driver")["interval_s"].shift(1)
    return laps


def add_quali_prior(laps: pd.DataFrame, quali: pd.DataFrame) -> pd.DataFrame:
    """Qualifying gap to pole in percent: a pace prior known before the race starts."""
    if quali.empty:
        laps["quali_gap_pct"] = np.nan
        return laps
    pole = quali["QualiBestS"].min()
    gap = (quali.set_index("Driver")["QualiBestS"] / pole - 1.0) * 100.0
    laps["quali_gap_pct"] = laps["driver"].map(gap)
    return laps


def is_wet_race(laps: pd.DataFrame) -> bool:
    return bool(laps["compound"].isin(WET_COMPOUNDS).any() or laps["rainfall"].eq(True).any())


def assign_exclusions(
    laps: pd.DataFrame,
    *,
    wet_race: bool,
    dry_only: bool,
    neutralised_status: list[str],
    slow_outlier_ratio: float,
) -> pd.Series:
    status = laps["track_status"].fillna("").astype(str)
    checks = {
        "wet_race": pd.Series(wet_race and dry_only, index=laps.index),
        "no_time": laps["lap_time_s"].isna(),
        "generated": laps["generated"].eq(True),
        "lap1": laps["lap_number"].eq(1),
        "pit_in": laps["pit_in_time_s"].notna(),
        "pit_out": laps["pit_out_time_s"].notna(),
        "neutralised": status.apply(lambda s: any(code in s for code in neutralised_status)),
        "non_slick": ~laps["compound"].isin(SLICKS),
        "inaccurate": laps["is_accurate"].ne(True),
    }
    reason = pd.Series(pd.NA, index=laps.index, dtype="string")
    for name, mask in checks.items():
        reason = reason.mask(reason.isna() & mask, name)

    # Spins, damage, long blue-flag laps: judged against the race's otherwise-clean laps.
    clean_median = laps.loc[reason.isna(), "lap_time_s"].median()
    if pd.notna(clean_median):
        slow = laps["lap_time_s"] > clean_median * slow_outlier_ratio
        reason = reason.mask(reason.isna() & slow, "slow_outlier")
    return reason


def build_race(race_path: Path, cfg: dict) -> tuple[pd.DataFrame, dict]:
    meta = json.loads((race_path / "meta.json").read_text(encoding="utf-8"))
    raw = pd.read_parquet(race_path / "laps.parquet")
    quali = pd.read_parquet(race_path / "quali.parquet")
    results = pd.read_parquet(race_path / "results.parquet")
    cleaning = cfg["cleaning"]

    laps = raw.rename(columns=RENAME).reindex(columns=list(RENAME.values()))
    laps["lap_number"] = laps["lap_number"].astype(int)
    total_laps = meta.get("total_laps") or meta["laps_completed"]

    laps = add_race_features(laps, total_laps)
    laps = add_quali_prior(laps, quali)
    grid = results.set_index("Abbreviation")["GridPosition"]
    laps["grid_position"] = laps["driver"].map(grid)

    race_id = f"{meta['year']}_{meta['round']:02d}"
    circuit = circuit_key(meta, cfg.get("circuit_aliases") or {})
    laps.insert(0, "race_id", race_id)
    laps.insert(1, "circuit", circuit)
    laps["team_season"] = laps["team"] + "_" + laps["year"].astype(str)

    wet = is_wet_race(laps)
    laps["exclude_reason"] = assign_exclusions(
        laps,
        wet_race=wet,
        dry_only=cfg["dry_only"],
        neutralised_status=[str(s) for s in cleaning["neutralised_track_status"]],
        slow_outlier_ratio=cleaning["slow_outlier_ratio"],
    )

    race = {
        "race_id": race_id,
        "year": meta["year"],
        "round": meta["round"],
        "event_name": meta["event_name"],
        "location": meta["location"],
        "circuit": circuit,
        "event_date": meta["event_date"],
        "total_laps": total_laps,
        "laps_completed": meta["laps_completed"],
        "wet": wet,
        "n_laps": len(laps),
        "n_clean": int(laps["exclude_reason"].isna().sum()),
    }
    return laps.reset_index(drop=True), race


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    raw_root = Path(cfg["paths"]["raw"])
    out_root = Path(cfg["paths"]["processed"])
    out_root.mkdir(parents=True, exist_ok=True)

    race_paths = sorted(p.parent for p in raw_root.glob("*/*/meta.json"))
    if not race_paths:
        log.error("no ingested races under %s; run strategist.data.ingest first", raw_root)
        return 1

    all_laps, races = [], []
    for path in race_paths:
        laps, race = build_race(path, cfg)
        all_laps.append(laps)
        races.append(race)

    laps = pd.concat(all_laps, ignore_index=True)
    races = pd.DataFrame(races)
    laps.to_parquet(out_root / "laps.parquet", index=False)
    races.to_parquet(out_root / "races.parquet", index=False)

    log.info("%d races, %d laps, %d clean (%.1f%%)", len(races), len(laps),
             races["n_clean"].sum(), 100 * races["n_clean"].sum() / len(laps))
    log.info("exclusions:\n%s", laps["exclude_reason"].value_counts().to_string())
    log.info("circuits: %s", ", ".join(sorted(races["circuit"].unique())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
