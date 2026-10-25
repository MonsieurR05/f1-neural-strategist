"""Lap-time forecast metrics.

Two families, because strategy decisions and absolute forecasts need different things:

- **absolute**: MAE / RMSE of the lap time, plus Gaussian NLL and interval coverage for the
  predicted uncertainty.
- **strategy-relevant**: comparing two strategies for the same driver in the same race cancels
  any constant error in that driver's race pace. ``within_mae`` removes each driver-race's mean
  error before scoring, and ``stint_slope_mae`` scores the per-stint degradation slope
  (lap time vs tyre age), which drives pit-stop timing.
"""

from statistics import NormalDist

import numpy as np
import pandas as pd

STINT_KEYS = ["race_id", "driver", "stint"]
DRIVER_RACE_KEYS = ["race_id", "driver"]


def mae(y: pd.Series, mu: pd.Series) -> float:
    return float(np.mean(np.abs(y - mu)))


def rmse(y: pd.Series, mu: pd.Series) -> float:
    return float(np.sqrt(np.mean((y - mu) ** 2)))


def gaussian_nll(y: pd.Series, mu: pd.Series, sigma: pd.Series) -> float:
    """Mean negative log-likelihood of y under N(mu, sigma²), in nats."""
    z = (y - mu) / sigma
    return float(np.mean(0.5 * z**2 + np.log(sigma) + 0.5 * np.log(2 * np.pi)))


def interval_coverage(y: pd.Series, mu: pd.Series, sigma: pd.Series, level: float = 0.9) -> float:
    """Share of y inside the central ``level`` interval; well calibrated ≈ ``level``."""
    half = NormalDist().inv_cdf(0.5 + level / 2) * sigma
    return float(np.mean(np.abs(y - mu) <= half))


def within_mae(df: pd.DataFrame, y: pd.Series, mu: pd.Series) -> float:
    """MAE after removing each driver-race's mean error (the error left for strategy comparison)."""
    err = (y - mu).groupby([df[k] for k in DRIVER_RACE_KEYS]).transform(lambda e: e - e.mean())
    return float(np.mean(np.abs(err)))


def stint_slopes(df: pd.DataFrame, values: pd.Series, min_laps: int = 8) -> pd.Series:
    """Least-squares slope of ``values`` against tyre age, per stint with enough clean laps."""
    frame = pd.DataFrame({"x": df["tyre_life"], "v": values, **{k: df[k] for k in STINT_KEYS}})

    def slope(g: pd.DataFrame) -> float:
        if len(g) < min_laps or g["x"].nunique() < 2:
            return np.nan
        x = g["x"] - g["x"].mean()
        return float((x * (g["v"] - g["v"].mean())).sum() / (x**2).sum())

    return frame.groupby(STINT_KEYS)[["x", "v"]].apply(slope).dropna()


def stint_slope_mae(df: pd.DataFrame, y: pd.Series, mu: pd.Series, min_laps: int = 8) -> tuple[float, int]:
    """MAE (s per lap of tyre age) between actual and predicted stint slopes, and the stint count."""
    actual = stint_slopes(df, y, min_laps)
    predicted = stint_slopes(df, mu, min_laps).reindex(actual.index)
    return float(np.mean(np.abs(actual - predicted))), len(actual)


def stint_slope_skill(df: pd.DataFrame, y: pd.Series, mu: pd.Series, min_laps: int = 8) -> float:
    """1 − slope MAE / slope MAE of predicting no net slope. 0 = no better than assuming fuel and
    tyre wear cancel within every stint; 1 = perfect per-stint slopes."""
    actual = stint_slopes(df, y, min_laps)
    model_err, _ = stint_slope_mae(df, y, mu, min_laps)
    return float(1.0 - model_err / np.mean(np.abs(actual)))


def evaluate(df: pd.DataFrame, y: pd.Series, mu: pd.Series, sigma: pd.Series) -> dict[str, float]:
    slope_mae, n_stints = stint_slope_mae(df, y, mu)
    return {
        "laps": len(df),
        "mae_s": mae(y, mu),
        "rmse_s": rmse(y, mu),
        "within_mae_s": within_mae(df, y, mu),
        "stint_slope_mae": slope_mae,
        "stint_slope_skill": stint_slope_skill(df, y, mu),
        "stints": n_stints,
        "nll": gaussian_nll(y, mu, sigma),
        "coverage_90": interval_coverage(y, mu, sigma, 0.9),
        "bias_s": float(np.mean(mu - y)),
    }
