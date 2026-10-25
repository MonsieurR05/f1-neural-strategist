"""Data source adapters.

Each adapter fetches one race into the same raw schema, so everything downstream
(``strategist.data.clean`` onwards) is source-agnostic:

- laps: FastF1 lap column names (``LAP_COLUMNS``), timedeltas as float seconds.
  ``Time`` / ``LapStartTime`` are seconds from a per-session origin; only differences
  within a race are meaningful.
- results: ``RESULT_COLUMNS``
- quali: ``Driver``, ``QualiBestS``
"""

from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

WEATHER_COLUMNS = ["AirTemp", "TrackTemp", "Humidity", "Pressure", "Rainfall", "WindSpeed"]

LAP_COLUMNS = [
    "Driver", "DriverNumber", "Team", "LapNumber", "LapTime", "Time", "LapStartTime",
    "Stint", "Compound", "TyreLife", "FreshTyre", "Position", "TrackStatus",
    "PitInTime", "PitOutTime", "Sector1Time", "Sector2Time", "Sector3Time",
    "IsAccurate", "FastF1Generated", *WEATHER_COLUMNS,
]

RESULT_COLUMNS = [
    "DriverNumber", "Abbreviation", "TeamName", "Position", "ClassifiedPosition",
    "GridPosition", "Status", "Points", "Laps", "Time",
]


class TimingUnavailableError(RuntimeError):
    """A source has no lap timing for a session (not a transient error)."""


@dataclass
class RaceData:
    laps: pd.DataFrame
    results: pd.DataFrame
    quali: pd.DataFrame
    meta: dict[str, Any]  # at least total_laps and laps_completed


class Source(Protocol):
    name: str

    def fetch(self, year: int, event: pd.Series) -> RaceData: ...


def timedeltas_to_seconds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="timedelta64[ns]").columns:
        df[col] = df[col].dt.total_seconds()
    return df


def make_source(name: str, cfg: dict) -> Source:
    jolpica_cache = cfg["paths"]["jolpica_cache"]
    if name == "fastf1":
        from strategist.data.sources.fastf1_source import FastF1Source
        return FastF1Source(jolpica_cache=jolpica_cache)
    if name == "openf1":
        from strategist.data.sources.openf1 import OpenF1Source
        return OpenF1Source(cache_dir=cfg["paths"]["openf1_cache"], jolpica_cache=jolpica_cache)
    raise ValueError(f"unknown data source {name!r}")
