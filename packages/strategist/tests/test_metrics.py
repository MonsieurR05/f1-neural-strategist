import numpy as np
import pandas as pd
import pytest

from strategist.eval.metrics import (
    evaluate,
    gaussian_nll,
    interval_coverage,
    mae,
    rmse,
    stint_slope_mae,
    within_mae,
)


def stint_frame(slope=0.1, n=10, drivers=("AAA", "BBB")) -> pd.DataFrame:
    rows = []
    for d in drivers:
        for age in range(1, n + 1):
            rows.append({"race_id": "2024_01", "driver": d, "stint": 1.0, "tyre_life": age,
                         "y": 90.0 + slope * age})
    return pd.DataFrame(rows)


def test_point_metrics():
    y, mu = pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 3.0, 1.0])
    assert mae(y, mu) == pytest.approx(1.0)
    assert rmse(y, mu) == pytest.approx(np.sqrt(5 / 3))


def test_gaussian_nll_standard_normal_at_mean():
    y = pd.Series([0.0])
    assert gaussian_nll(y, y, pd.Series([1.0])) == pytest.approx(0.5 * np.log(2 * np.pi))


def test_coverage_counts_inside_interval():
    y = pd.Series([0.0, 1.0, 2.0, 10.0])
    cov = interval_coverage(y, pd.Series([0.0] * 4), pd.Series([1.0] * 4), level=0.9)  # ±1.645
    assert cov == pytest.approx(0.5)


def test_within_mae_ignores_constant_driver_race_offset():
    df = stint_frame()
    mu = df["y"] + np.where(df["driver"] == "AAA", 1.5, -0.7)  # constant pace error per driver
    assert mae(df["y"], mu) > 1.0
    assert within_mae(df, df["y"], mu) == pytest.approx(0.0, abs=1e-12)


def test_stint_slope_mae():
    df = stint_frame(slope=0.1)
    flat = pd.Series(90.0, index=df.index)
    err, n = stint_slope_mae(df, df["y"], flat)
    assert n == 2 and err == pytest.approx(0.1)
    err, _ = stint_slope_mae(df, df["y"], df["y"] + 3.0)  # offset doesn't change slopes
    assert err == pytest.approx(0.0, abs=1e-12)


def test_short_stints_are_not_scored():
    df = stint_frame(n=5)
    _, n = stint_slope_mae(df, df["y"], df["y"])
    assert n == 0


def test_evaluate_keys():
    df = stint_frame()
    out = evaluate(df, df["y"], df["y"] + 0.1, pd.Series(0.5, index=df.index))
    assert out["bias_s"] == pytest.approx(0.1)
    assert out["within_mae_s"] == pytest.approx(0.0, abs=1e-12)
    assert set(out) >= {"mae_s", "rmse_s", "nll", "coverage_90", "stint_slope_mae"}
