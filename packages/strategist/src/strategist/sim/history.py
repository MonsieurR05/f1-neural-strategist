"""Build simulator inputs from a historical race: what was actually raced.

Used by the backtest (replay a race with the strategies, neutralisations and retirements
that happened) and later by the app as the starting point for what-if edits.
"""

import re

import numpy as np
import pandas as pd

from strategist.models.features import COMPOUNDS
from strategist.sim.params import lap_status
from strategist.sim.race import Car, RaceSetup, Stint, Strategy

FINISHED = re.compile(r"Finished|Lap|^\+")


def strategy_from_laps(driver_laps: pd.DataFrame) -> Strategy:
    """Stints in lap order: compound, last lap on the set, tyre age when fitted."""
    stints = []
    for _, g in driver_laps.sort_values("lap_number").groupby("stint", sort=True):
        compound = g["compound"].dropna()
        compound = compound.iat[0] if len(compound) and compound.iat[0] in COMPOUNDS else "MEDIUM"
        first_age = g["tyre_life"].dropna()
        age0 = int(max(first_age.iat[0] - 1, 0)) if len(first_age) else 0
        stints.append(Stint(compound, int(g["lap_number"].max()), age0))
    return Strategy(stints or [Stint("MEDIUM", 1)])


def nominated_hardness(laps: pd.DataFrame) -> dict[str, float] | None:
    """Hardness of the race's nominated hard/medium/soft (``None`` if not in the table)."""
    cols = [f"h_{c}" for c in COMPOUNDS]
    if not set(cols) <= set(laps.columns) or laps[cols].iloc[0].isna().any():
        return None
    return {c: float(laps[f"h_{c}"].iloc[0]) for c in COMPOUNDS}


def finishers(results: pd.DataFrame) -> set[str]:
    status = results["Status"].fillna("").astype(str)
    return set(results.loc[status.str.contains(FINISHED), "Abbreviation"])


def setup_from_history(race_laps: pd.DataFrame, results: pd.DataFrame,
                       start_lap: int = 1) -> tuple[RaceSetup, pd.DataFrame]:
    """Replay setup starting after ``start_lap`` laps, plus the actual outcome to compare with.

    Cars without a time at ``start_lap`` (lap-1 retirements) are left out. Actual finishing
    order is road order at the line (laps completed, then time), not the official
    classification, which includes post-race penalties.
    """
    laps = race_laps.sort_values(["driver", "lap_number"])
    total = int(laps["lap_number"].max())
    at_start = laps[laps["lap_number"] == start_lap].dropna(subset=["session_time_s"])
    drivers = at_start.sort_values("session_time_s")["driver"].tolist()
    finished = finishers(results)

    cars, retirements = [], {}
    for d in drivers:
        dl = laps[laps["driver"] == d]
        first = dl.iloc[0]
        cars.append(Car(
            driver=d, q_s=float(first["q_s"]), quali_gap_pct=float(first["quali_gap_pct"]),
            quali_missing=float(first["quali_missing"]),
            grid_position=float(first["grid_position"]) if pd.notna(first["grid_position"]) else 20.0,
            strategy=strategy_from_laps(dl), team=str(first["team"])))
        if d not in finished:  # a non-finisher never completes the full distance
            last = int(dl.dropna(subset=["session_time_s"])["lap_number"].max())
            retirements[d] = min(last, total - 1)

    start = at_start.set_index("driver").loc[drivers, "session_time_s"].to_numpy()
    status = lap_status(laps).droplevel(0)
    events = {int(lap): s for lap, s in status.items() if s != "GREEN" and lap > start_lap}
    setup = RaceSetup(
        circuit=str(laps["circuit"].iat[0]), total_laps=total,
        track_temp=float(laps["track_temp"].median()), cars=cars, start_lap=start_lap,
        start_times=start - start.min(), events=events, retirements=retirements,
        hardness=nominated_hardness(laps))

    # actual road order at the finish and positions per lap
    last = laps.dropna(subset=["session_time_s"]).groupby("driver").agg(
        laps_done=("lap_number", "max"), t=("session_time_s", "last"))
    last = last.loc[drivers]
    last["finished"] = last.index.isin(finished)
    ranked = last.sort_values(["finished", "laps_done", "t"], ascending=[False, False, True])
    ranked["actual_pos"] = np.arange(1, len(ranked) + 1)
    winner_t = ranked.loc[ranked["finished"], "t"].min()
    ranked["actual_gap_s"] = np.where(ranked["laps_done"] == total, ranked["t"] - winner_t, np.nan)
    per_lap = laps.pivot_table(index="lap_number", columns="driver", values="position")
    outcome = ranked.reset_index().rename(columns={"index": "driver"})
    outcome.attrs["positions_by_lap"] = per_lap.reindex(columns=drivers)
    return setup, outcome
