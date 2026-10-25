import numpy as np
import pandas as pd
import pytest

from strategist.models.baselines import CircuitPace, FuelDegLinear, LightGBMLapTime
from strategist.models.features import GAP_CAP_S, prepare

FUEL = 0.0008   # ratio per lap of fuel on board
DEG = {"SOFT": 0.0015, "MEDIUM": 0.0010, "HARD": 0.0007}
OFFSET = {"SOFT": 0.0, "MEDIUM": 0.004, "HARD": 0.008}


def synthetic(circuits=("A", "B"), years=(2021, 2022, 2023), noise=0.0, seed=0) -> pd.DataFrame:
    """Two-stop races on known fuel/deg/offset parameters (ratio units, q_s = 90 s)."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        for c_i, circuit in enumerate(circuits):
            base = 1.05 + 0.02 * c_i
            for d in range(6):
                plan = [("SOFT", 1, 15), ("MEDIUM", 16, 35), ("HARD", 36, 50)]
                if d % 2:
                    plan = [("MEDIUM", 1, 20), ("HARD", 21, 38), ("SOFT", 39, 50)]
                for stint, (cmp, a, b) in enumerate(plan, start=1):
                    for lap in range(a, b + 1):
                        age = lap - a + 1
                        ratio = (base + FUEL * (50 - lap) + OFFSET[cmp] + DEG[cmp] * age
                                 + noise * rng.standard_normal())
                        rows.append({
                            "race_id": f"{year}_{c_i}", "year": year, "circuit": circuit,
                            "driver": f"D{d}", "lap_number": lap, "laps_remaining": 50 - lap,
                            "stint": float(stint), "compound": cmp, "tyre_life": float(age),
                            "q_s": 90.0, "lap_time_s": 90.0 * ratio, "ratio": ratio,
                        })
    return pd.DataFrame(rows)


def test_circuit_pace_uses_circuit_ratio_and_global_fallback():
    df = synthetic()
    m = CircuitPace().fit(df)
    unseen = df.head(3).assign(circuit="Z")
    assert np.allclose(m.predict(unseen), m.ratio_global * 90.0)
    assert m.ratio_by_circuit["B"] > m.ratio_by_circuit["A"]


def test_fuel_deg_linear_recovers_parameters():
    df = synthetic()
    m = FuelDegLinear(prior_laps=0.0).fit(df)
    c = m.coefficients().loc["A"]
    assert c["laps_remaining"] == pytest.approx(FUEL, abs=1e-9)
    assert c["age_SOFT"] == pytest.approx(DEG["SOFT"], abs=1e-9)
    assert c["is_HARD"] == pytest.approx(OFFSET["HARD"], abs=1e-9)
    assert np.allclose(m.predict(df), df["lap_time_s"])


def test_fuel_deg_linear_shrinks_missing_compound_to_pooled():
    full = synthetic(circuits=("A",))
    no_hard = synthetic(circuits=("B",))
    no_hard = no_hard[no_hard["compound"] != "HARD"]
    m = FuelDegLinear(prior_laps=300).fit(pd.concat([full, no_hard]))
    # B never ran the hard: its hard coefficients come from the pooled fit, not zero
    assert m.coefficients().loc["B", "age_HARD"] == pytest.approx(DEG["HARD"], rel=0.05)


def test_unseen_circuit_uses_pooled_coefficients():
    df = synthetic()
    m = FuelDegLinear().fit(df)
    pred = m.predict(df.head(5).assign(circuit="Z"))
    assert pred.notna().all()


def test_lightgbm_fits_and_predicts_close_on_noiseless_data():
    df = synthetic(noise=0.0005)
    for col, val in {"fresh_tyre": 1.0, "fuel_frac": 0.5, "total_laps": 50.0,
                     "quali_gap_pct": 1.0, "quali_missing": 0.0, "grid_position": 5.0,
                     "track_temp": 30.0, "air_temp": 25.0, "humidity": 40.0,
                     "gap_ahead_s": GAP_CAP_S, "is_leader": 0.0, "hardness": 3.0}.items():
        df[col] = val
    m = LightGBMLapTime({"min_data_in_leaf": 5, "learning_rate": 0.1}, max_rounds=400,
                        early_stopping_rounds=50).fit(df)
    err = np.abs(m.predict(df) - df["lap_time_s"]).mean()
    assert err < 0.15 and m.rounds > 0


def test_prepare_fills_missing_quali_and_leader_gap():
    laps = pd.DataFrame({
        "race_id": ["r", "r", "r"], "driver": ["A", "B", "C"],
        "quali_best_s": [90.0, 91.0, np.nan], "quali_pole_s": [90.0] * 3,
        "quali_gap_pct": [0.0, 1.111, np.nan], "lap_time_s": [95.0, 96.0, 97.0],
        "interval_start_s": [np.nan, 1.2, 25.0], "fresh_tyre": [True, False, True],
        "circuit": ["X"] * 3, "compound": ["SOFT"] * 3,
    })
    out = prepare(laps).set_index("driver")
    assert out.loc["C", "q_s"] == 91.0 and out.loc["C", "quali_missing"] == 1.0
    assert out.loc["A", "is_leader"] == 1.0 and out.loc["A", "gap_ahead_s"] == GAP_CAP_S
    assert out.loc["C", "gap_ahead_s"] == GAP_CAP_S
    assert out.loc["B", "ratio"] == pytest.approx(96.0 / 91.0)
