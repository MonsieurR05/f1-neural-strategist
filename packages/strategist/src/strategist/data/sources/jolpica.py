"""Race classification and qualifying times from Jolpica (the Ergast successor), via FastF1."""

import pandas as pd
from fastf1.ergast import Ergast

from strategist.data.sources import RESULT_COLUMNS

_ergast = Ergast(result_type="pandas", auto_cast=True)


def race_results(year: int, rnd: int) -> pd.DataFrame:
    r = _ergast.get_race_results(season=year, round=rnd).content[0]
    return pd.DataFrame({
        "DriverNumber": r["number"].astype(str),
        "Abbreviation": r["driverCode"],
        "TeamName": r["constructorName"],
        "Position": r["position"].astype(float),
        "ClassifiedPosition": r["positionText"].astype(str),
        "GridPosition": r["grid"].astype(float),
        "Status": r["status"],
        "Points": r["points"].astype(float),
        "Laps": r["laps"].astype(float),
        "Time": r["totalRaceTime"].dt.total_seconds(),
    }).reindex(columns=RESULT_COLUMNS)


def quali_best(year: int, rnd: int) -> pd.DataFrame:
    """Best of Q1-Q3 per driver. Empty when Jolpica has the order but no times (e.g. 2025 R06)."""
    q = _ergast.get_qualifying_results(season=year, round=rnd).content[0]
    sessions = q.reindex(columns=["Q1", "Q2", "Q3"]).apply(
        lambda s: pd.to_timedelta(s, errors="coerce").dt.total_seconds()
    )
    return pd.DataFrame({"Driver": q["driverCode"], "QualiBestS": sessions.min(axis=1)}).dropna()
