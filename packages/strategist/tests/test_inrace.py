import numpy as np
import pandas as pd
import pytest

from strategist.models.inrace import GP_PARAMS, InRaceGP, InRaceGRU, green_flag_laps, known_at_cutoff


def synthetic_race(race_id="2024_01", drivers=8, laps=40, pit=20, seed=0, offsets=None,
                   hard_wear=0.0, noise=0.1):
    """Clean laps for one race: every driver soft to lap ``pit``, hard after (except driver 0,
    who starts on the hard and stays out). Residual = driver offset + hard wear error × age."""
    rng = np.random.default_rng(seed)
    offsets = rng.normal(0, 1, drivers) if offsets is None else np.asarray(offsets)
    rows = []
    for d in range(drivers):
        for lap in range(2, laps + 1):
            first_hard = d == 0
            on_hard = first_hard or lap > pit
            stint = 1 if (first_hard or lap <= pit) else 2
            age = lap if (first_hard or lap <= pit) else lap - pit
            r = offsets[d] + (hard_wear * age / 10 if on_hard else 0.0) + rng.normal(0, noise)
            rows.append({"race_id": race_id, "driver": f"D{d}", "stint": stint, "lap_number": lap,
                         "tyre_life": float(age), "compound": "HARD" if on_hard else "SOFT",
                         "hardness": 2.0 if on_hard else 3.0, "total_laps": laps, "q_s": 90.0,
                         "since_green": lap - 1, "green_period": 0,
                         "resid": r})
    return pd.DataFrame(rows)


def matched_gp(**overrides):
    """GP whose hyperparameters match the synthetic process (driver offsets ~N(0, 1), small noise)."""
    state = GP_PARAMS | {"driver": 1.0, "noise": 0.1, "swing": 0.05, "mean": 0.0, "mean_wear": 0.0}
    return InRaceGP.from_state(state | overrides)


def split(frame, cutoff):
    return frame[frame["lap_number"] <= cutoff], frame[frame["lap_number"] > cutoff]


def test_gp_learns_driver_offsets_and_shrinks_uncertainty():
    race = synthetic_race(offsets=[2.0, -1.0, 0.5, 0.0, 1.0, -0.5, 0.3, -2.0])
    obs, target = split(race, 15)
    gp = matched_gp()
    mu, sigma = gp.predict(obs, target)
    _, prior_sigma = gp.predict(obs.iloc[:0], target)
    by_driver = pd.Series(mu, index=target.index).groupby(target["driver"]).mean()
    truth = target.groupby("driver")["resid"].mean()
    assert np.abs(by_driver - truth).max() < 0.3  # vs up to 2.0 with no update
    assert (sigma < prior_sigma).all()


def test_green_flag_laps_restart_after_safety_car():
    laps = pd.DataFrame({"race_id": "2025_01", "lap_number": range(1, 11), "position": 1,
                         "track_status": ["1", "1", "4", "4", "1", "1", "1", "1", "1", "1"]})
    flags = green_flag_laps(laps)
    by_lap = flags.set_index("lap_number")
    assert by_lap["since_green"].tolist() == [0, 1, 0, 0, 1, 2, 3, 4, 5, 6]
    assert by_lap.loc[2, "green_period"] < by_lap.loc[5, "green_period"]
    future = known_at_cutoff(pd.DataFrame({"lap_number": [8, 9]}), flags, 7)
    assert future["since_green"].tolist() == [4, 5]
    under_sc = known_at_cutoff(pd.DataFrame({"lap_number": [5, 6]}), flags, 4)
    assert under_sc["since_green"].tolist() == [1, 2]
    assert (under_sc["green_period"] == by_lap.loc[5, "green_period"]).all()


def test_gp_prior_without_observations():
    race = synthetic_race()
    gp = InRaceGP()
    mu, sigma = gp.predict(race.iloc[:0], race)
    assert np.allclose(mu, 0.0) and (sigma > 1.0).all()


def test_gp_carries_compound_wear_across_drivers():
    """Driver 0 has run 20 laps on a hard that wears faster than lap_net expected; the others
    haven't fitted it yet, but their future hard stints should inherit the extra wear."""
    worn = synthetic_race(offsets=np.zeros(8), hard_wear=1.5, noise=0.05)
    flat = synthetic_race(offsets=np.zeros(8), hard_wear=0.0, noise=0.05)
    gp = InRaceGP()
    preds = []
    for race in (worn, flat):
        obs, target = split(race, 20)
        mu, _ = gp.predict(obs, target)
        late_hard = (target["driver"] != "D0") & (target["tyre_life"] >= 15)
        preds.append(mu[late_hard.to_numpy()].mean())
    assert preds[0] > preds[1] + 0.3


@pytest.mark.parametrize("model", ["gp", "gru"])
def test_forecasts_ignore_target_residuals(model):
    train = pd.concat([synthetic_race(f"2023_{i:02d}", seed=i) for i in range(3)]
                      + [synthetic_race(f"2024_{i:02d}", seed=10 + i) for i in range(2)])
    m = InRaceGP(steps=5) if model == "gp" else InRaceGRU(hidden=8, max_epochs=2, patience=1)
    m.fit(train)
    race = synthetic_race("2025_01", seed=99)
    obs, target = split(race, 18)
    mu1, s1 = m.predict(obs, target)
    mu2, s2 = m.predict(obs, target.assign(resid=1e3))
    mu3, s3 = m.predict(obs, target.drop(columns="resid"))
    assert len(mu1) == len(target) and np.isfinite(mu1).all() and (s1 > 0).all()
    assert np.allclose(mu1, mu2) and np.allclose(mu1, mu3) and np.allclose(s1, s2)


def test_gru_uses_observed_laps():
    train = pd.concat([synthetic_race(f"2023_{i:02d}", seed=i) for i in range(6)]
                      + [synthetic_race(f"2024_{i:02d}", seed=10 + i) for i in range(3)])
    gru = InRaceGRU(hidden=16, max_epochs=40, patience=40).fit(train)
    race = synthetic_race("2025_01", offsets=[3.0, -3.0, 0, 0, 0, 0, 0, 0], seed=5)
    obs, target = split(race, 20)
    mu, _ = gru.predict(obs, target)
    by_driver = pd.Series(mu, index=target.index).groupby(target["driver"]).mean()
    assert by_driver["D0"] > by_driver["D1"] + 2.0
