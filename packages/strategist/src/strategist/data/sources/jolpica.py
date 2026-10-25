"""Race classification and qualifying times from Jolpica (the Ergast successor), via FastF1.

Results for past races don't change, so they are cached as Parquet and Jolpica is only
asked once per race.
"""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
from fastf1.ergast import Ergast

from strategist.data.sources import RESULT_COLUMNS

_ergast = Ergast(result_type="pandas", auto_cast=True)


def _cached(cache_dir: str | Path, name: str, fetch: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    path = Path(cache_dir) / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def race_results(year: int, rnd: int, cache_dir: str | Path) -> pd.DataFrame:
    return _cached(cache_dir, f"{year}_{rnd:02d}_results", lambda: _race_results(year, rnd))


def quali_best(year: int, rnd: int, cache_dir: str | Path) -> pd.DataFrame:
    """Best of Q1-Q3 per driver. Empty when Jolpica has the order but no times (e.g. 2025 R06)."""
    return _cached(cache_dir, f"{year}_{rnd:02d}_quali", lambda: _quali_best(year, rnd))


def _race_results(year: int, rnd: int) -> pd.DataFrame:
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


def _quali_best(year: int, rnd: int) -> pd.DataFrame:
    q = _ergast.get_qualifying_results(season=year, round=rnd).content[0]
    sessions = q.reindex(columns=["Q1", "Q2", "Q3"]).apply(
        lambda s: pd.to_timedelta(s, errors="coerce").dt.total_seconds()
    )
    return pd.DataFrame({"Driver": q["driverCode"], "QualiBestS": sessions.min(axis=1)}).dropna()
