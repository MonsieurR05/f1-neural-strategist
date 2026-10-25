import numpy as np
import pandas as pd
import pytest
import torch

from strategist.models.features import GAP_CAP_S
from strategist.models.lapnet import CONTEXT_FEATURES, Batch, LapNetModule, LapTimeNet
from tests.test_baselines import DEG, FUEL, synthetic


def with_inputs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, val in {"fresh_tyre": 1.0, "fuel_frac": 0.5, "total_laps": 50.0,
                     "quali_gap_pct": 1.0, "quali_missing": 0.0, "grid_position": 5.0,
                     "track_temp": 30.0, "gap_ahead_s": GAP_CAP_S, "is_leader": 0.0}.items():
        df[col] = val
    df["fuel_frac"] = df["laps_remaining"] / 50.0
    return df


def random_batch(n: int, ages: torch.Tensor) -> Batch:
    g = torch.Generator().manual_seed(0)
    return Batch(circuit=torch.randint(0, 5, (n,), generator=g),
                 compound=torch.randint(0, 3, (n,), generator=g),
                 context=torch.randn(n, len(CONTEXT_FEATURES), generator=g),
                 tyre_life=ages, laps_remaining=torch.full((n,), 20.0), fresh=torch.ones(n),
                 gap=torch.full((n,), 3.0), leader=torch.zeros(n), fuel_frac=torch.full((n,), 0.4),
                 hardness=torch.full((n,), 3.0))


def test_degradation_is_non_decreasing_in_tyre_age_for_any_context():
    torch.manual_seed(1)
    module = LapNetModule(n_circuits=5)
    for p in module.parameters():  # randomise heavily: the guarantee must not depend on training
        torch.nn.init.normal_(p, std=1.0)
    ages = torch.arange(0, 80, dtype=torch.float32)
    contexts = [random_batch(1, torch.tensor([0.0])) for _ in range(20)]
    for ctx in contexts:
        b = Batch(**{k: (v.repeat(len(ages)) if v.dim() == 1 else v.repeat(len(ages), 1))
                     for k, v in ctx.__dict__.items()})
        b.tyre_life = ages
        deg = module(b)["deg"]
        assert torch.all(deg[1:] - deg[:-1] >= -1e-5)


def test_full_prediction_is_non_decreasing_in_tyre_age():
    """The interaction term doesn't see tyre age, so the guarantee holds for mu itself."""
    torch.manual_seed(2)
    module = LapNetModule(n_circuits=5, interaction=True)
    for p in module.parameters():
        torch.nn.init.normal_(p, std=1.0)
    ages = torch.arange(0, 80, dtype=torch.float32)
    ctx = random_batch(1, torch.tensor([0.0]))
    b = Batch(**{k: (v.repeat(len(ages)) if v.dim() == 1 else v.repeat(len(ages), 1))
                 for k, v in ctx.__dict__.items()})
    b.tyre_life = ages
    mu = module(b)["mu"]
    assert torch.all(mu[1:] - mu[:-1] >= -1e-5)


def test_softer_compounds_are_faster_fresh_and_wear_more_for_any_weights():
    torch.manual_seed(5)
    module = LapNetModule(n_circuits=5)
    for p in module.parameters():
        torch.nn.init.normal_(p, std=1.0)
    ages = torch.arange(0, 60, dtype=torch.float32)
    for circuit in range(5):
        for fresh in (0.0, 1.0):
            base = random_batch(1, torch.tensor([0.0]))
            mus = {}
            for k in range(3):  # SOFT, MEDIUM, HARD
                b = Batch(**{n: (v.repeat(len(ages)) if v.dim() == 1 else v.repeat(len(ages), 1))
                             for n, v in base.__dict__.items()})
                b.tyre_life, b.circuit = ages, torch.full((len(ages),), circuit)
                b.compound, b.fresh = torch.full((len(ages),), k), torch.full((len(ages),), fresh)
                b.hardness = torch.full((len(ages),), 4.0 - k)  # soft 4 > medium 3 > hard 2
                out = module(b)
                mus[k] = (out["offset"], out["deg"])
            # fresh pace: soft ≤ medium ≤ hard; wear: soft ≥ medium ≥ hard at every age
            assert torch.all(mus[0][0] <= mus[1][0] + 1e-5) and torch.all(mus[1][0] <= mus[2][0] + 1e-5)
            assert torch.all(mus[0][1] >= mus[1][1] - 1e-5) and torch.all(mus[1][1] >= mus[2][1] - 1e-5)


def test_sigma_is_positive_and_floored():
    module = LapNetModule(n_circuits=5)
    out = module(random_batch(64, torch.rand(64) * 40))
    assert torch.all(out["sigma"] >= 0.05)


@pytest.fixture(scope="module")
def fitted() -> tuple[LapTimeNet, pd.DataFrame]:
    df = with_inputs(synthetic(noise=0.0003, seed=3))
    # small batches: the synthetic set has only ~1,800 laps
    net = LapTimeNet(max_epochs=40, patience=40, lr=1e-2, batch_size=128, sigma_epochs=3,
                     threads=4)
    return net.fit(df), df


def test_recovers_fuel_and_degradation_on_synthetic_races(fitted):
    net, df = fitted
    comps = net.components(df)
    # fuel term is linear in laps remaining: its slope is the fuel effect (s per lap of fuel)
    fuel_slope = np.polyfit(df["laps_remaining"], comps["fuel"], 1)[0]
    assert fuel_slope == pytest.approx(FUEL * 90.0, rel=0.35)
    soft = df["compound"] == "SOFT"
    deg_slope = np.polyfit(df.loc[soft, "tyre_life"], comps.loc[soft, "deg"], 1)[0]
    assert deg_slope == pytest.approx(DEG["SOFT"] * 90.0, rel=0.35)
    err = np.abs(net.predict(df) - df["lap_time_s"]).mean()
    assert err < 0.1


def test_save_load_roundtrip(fitted, tmp_path):
    net, df = fitted
    net.save(tmp_path / "net.pt")
    loaded = LapTimeNet.load(tmp_path / "net.pt")
    a, b = net.predict_dist(df.head(20)), loaded.predict_dist(df.head(20))
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
    assert len(loaded.oos_residuals) == len(df)


def test_predict_dist_and_unseen_circuit(fitted):
    net, df = fitted
    mu, sigma = net.predict_dist(df.head(10).assign(circuit="Nowhere"))
    assert mu.notna().all() and (sigma > 0).all()
