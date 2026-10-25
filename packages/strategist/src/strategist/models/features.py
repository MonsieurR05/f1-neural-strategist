"""Model inputs derived from the cleaned lap table. Shared by every lap-time model.

All inputs are known before the lap starts: pre-race information (qualifying, grid, circuit,
race distance) and per-lap state (tyres, lap number, weather sample, gap to the car ahead
at the line).
"""

import numpy as np
import pandas as pd

GAP_CAP_S = 10.0  # beyond this the car ahead no longer matters; leaders get the cap

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

NUMERIC_FEATURES = [
    "tyre_life", "fresh_tyre", "stint", "lap_number", "laps_remaining", "fuel_frac",
    "total_laps", "quali_gap_pct", "quali_missing", "grid_position", "track_temp",
    "air_temp", "humidity", "gap_ahead_s", "is_leader", "hardness",
]
CATEGORICAL_FEATURES = ["circuit", "compound"]


def clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Laps usable for lap-time modelling: no exclusion reason, tyre age known, and from a race
    whose qualifying was representative (the pace anchor ``q_s`` is meaningful)."""
    usable = laps["exclude_reason"].isna() & laps["quali_representative"]
    return laps[usable].dropna(subset=["tyre_life", "lap_time_s"])


def prepare(laps: pd.DataFrame) -> pd.DataFrame:
    """Add the pace anchor ``q_s``, the normalised target ``ratio`` and traffic inputs.

    ``q_s`` is the driver's best qualifying lap: it carries the car/driver pace and the
    circuit-season pace level, so models predict ``ratio = lap_time / q_s`` and transfer
    across seasons. Drivers without a qualifying time get the slowest time in their race
    and ``quali_missing = 1``.
    """
    df = laps.copy()
    slowest = df.groupby("race_id")["quali_best_s"].transform("max")
    df["quali_missing"] = df["quali_best_s"].isna().astype(float)
    df["q_s"] = df["quali_best_s"].fillna(slowest)
    df["quali_gap_pct"] = df["quali_gap_pct"].fillna(
        (df["q_s"] / df["quali_pole_s"] - 1.0) * 100.0)
    df = df.dropna(subset=["q_s"])
    df["ratio"] = df["lap_time_s"] / df["q_s"]

    df["is_leader"] = df["interval_start_s"].isna().astype(float)
    df["gap_ahead_s"] = df["interval_start_s"].fillna(GAP_CAP_S).clip(upper=GAP_CAP_S)
    df["fresh_tyre"] = df["fresh_tyre"].astype(float)
    if "hardness" not in df:  # Pirelli compound hardness (data.compounds); NaN if unknown
        df["hardness"] = np.nan
    for col in ("circuit", "compound"):
        df[col] = df[col].astype(str)
    return df


def season_mask(df: pd.DataFrame, seasons: list[int]) -> pd.Series:
    return df["year"].isin(seasons)


def as_float(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].astype(float).to_numpy()
