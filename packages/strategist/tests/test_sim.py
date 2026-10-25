"""Simulator mechanics on a stub lap model with known, deterministic behaviour."""

import numpy as np
import pandas as pd
import pytest
import torch

from strategist.sim.params import SimParams
from strategist.sim.race import Car, RaceSetup, Simulator, Stint, Strategy

Q = 90.0


class StubNet:
    """Lap time = q · (1 + pace_pct/100) per car; σ tiny. Pace differences are set via
    quali_gap_pct, which lands in context column 0 (unstandardised here)."""

    def __init__(self, sigma_pct: float = 1e-4, deg_pct_per_lap: float = 0.0):
        self.sigma_pct, self.deg = sigma_pct, deg_pct_per_lap

    def static_inputs(self, df):
        ctx = torch.zeros(len(df), 5)
        ctx[:, 0] = torch.tensor(df["quali_gap_pct"].to_numpy(), dtype=torch.float32)
        return torch.zeros(len(df), dtype=torch.long), ctx

    def predict_batch(self, b):
        mu = 5.0 + b.context[:, 0] + self.deg * b.tyre_life
        return mu, torch.full_like(mu, self.sigma_pct)


def params(**kw) -> SimParams:
    base = dict(pit_loss={}, pit_loss_global=20.0, pit_loss_sd=0.0, pit_loss_neutral_ratio=0.5,
                sc_lap_ratio=1.5, vsc_lap_ratio=1.35, sc_queue_gap=0.8, sc_compress=0.35,
                overtake_intercept={}, overtake_intercept_global=-50.0, overtake_pace=0.0,
                overtake_gap=0.0, held_gap_low=0.5, held_gap_high=0.5, common_var_share=0.0,
                driver_var_share=0.0)
    return SimParams(**{**base, **kw})


def car(name, pace_pct, stints=None) -> Car:
    return Car(name, Q, pace_pct, 0.0, 1.0, Strategy(stints or [Stint("MEDIUM", 99)]))


def setup(cars, gaps, laps=10, **kw) -> RaceSetup:
    return RaceSetup("X", laps, 30.0, cars, start_lap=1, start_times=np.array(gaps, float), **kw)


def test_order_holds_when_overtaking_is_impossible():
    # B is 1 s/lap faster but starts 0.5 s behind: with P(pass) ≈ 0 it stays held behind A
    s = setup([car("A", 1.0), car("B", 0.0)], [0.0, 0.5])
    res = Simulator(StubNet(), params(), runs=20).run(s)
    assert (res.positions()[:, 1] == 2).all()
    gap = res.times[-1][:, 1] - res.times[-1][:, 0]
    assert np.allclose(gap, 0.5)


def test_without_interactions_the_faster_car_gets_through():
    s = setup([car("A", 1.0), car("B", 0.0)], [0.0, 0.5])
    res = Simulator(StubNet(), params(), runs=20, interactions=False).run(s)
    assert (res.positions()[:, 1] == 1).all()


def test_certain_pass_happens():
    s = setup([car("A", 1.0), car("B", 0.0)], [0.0, 0.5])
    res = Simulator(StubNet(), params(overtake_intercept_global=50.0), runs=20).run(s)
    assert (res.positions()[:, 1] == 1).all()


def test_pit_stop_costs_the_pit_loss():
    stop = [Stint("MEDIUM", 5), Stint("HARD", 99)]
    s = setup([car("A", 0.0), car("B", 0.0, stop)], [0.0, 100.0])  # far apart: no interaction
    res = Simulator(StubNet(), params(), runs=5).run(s)
    gap = res.times[-1][:, 1] - res.times[-1][:, 0]
    assert np.allclose(gap, 100.0 + 20.0)


def test_safety_car_closes_up_the_field_and_vsc_keeps_gaps():
    cars = [car("A", 0.0), car("B", 0.0), car("C", 0.0)]
    # gaps shrink to 35% per SC lap: 10 s and 20 s gaps are a queue after three laps
    sc = Simulator(StubNet(), params(), runs=3).run(setup(cars, [0, 10, 30], events={3: "SC", 4: "SC", 5: "SC"}))
    gaps = np.diff(sc.times[5][0])
    assert np.all(gaps < 1.0)
    vsc = Simulator(StubNet(), params(), runs=3).run(setup(cars, [0, 10, 30], events={3: "VSC"}))
    assert np.allclose(np.diff(vsc.times[3][0]), [10, 20])


def test_pitting_under_safety_car_is_cheaper():
    stop = [Stint("MEDIUM", 3), Stint("HARD", 99)]
    cars = [car("A", 0.0), car("B", 0.0, stop)]
    green = Simulator(StubNet(), params(), runs=2).run(setup(cars, [0, 100]))
    under_sc = Simulator(StubNet(), params(), runs=2).run(setup(cars, [0, 100], events={3: "VSC"}))
    lost = lambda r: (r.times[3][0, 1] - r.times[3][0, 0]) - (r.times[2][0, 1] - r.times[2][0, 0])
    assert lost(under_sc) == pytest.approx(10.0, abs=1e-2)
    assert lost(green) == pytest.approx(20.0, abs=1e-2)


def test_tyre_plan_resets_age_after_a_stop():
    comp, age, fresh = Strategy([Stint("SOFT", 3, 2), Stint("HARD", 99)]).tyre_plan(6)
    assert age.tolist() == [3, 4, 5, 1, 2, 3]
    assert fresh.tolist() == [0, 0, 0, 1, 1, 1]
    assert comp.tolist() == [0, 0, 0, 2, 2, 2]


def test_retirement_and_reproducibility():
    cars = [car("A", 0.0), car("B", 0.0)]
    s = setup(cars, [0, 50], retirements={"A": 4})
    r1 = Simulator(StubNet(sigma_pct=0.5), params(), runs=10, seed=7).run(s)
    r2 = Simulator(StubNet(sigma_pct=0.5), params(), runs=10, seed=7).run(s)
    assert np.array_equal(r1.times, r2.times)
    assert np.isinf(r1.times[5][:, 0]).all() and (r1.positions()[:, 0] == 2).all()


def test_driver_offset_widens_outcomes_but_common_shift_does_not():
    cars = [car(n, 0.0) for n in "ABCD"]
    s = setup(cars, [0, 2, 4, 6], laps=40)

    def win_odds_spread(**shares):
        res = Simulator(StubNet(sigma_pct=0.5), params(**shares), runs=300,
                        interactions=False).run(s)
        return res.finish_table()["p_win"].std()

    # a race-long per-driver offset reshuffles the order between runs → flatter win odds
    assert win_odds_spread(driver_var_share=0.5) < win_odds_spread()
    # the same variance shared by the whole field barely changes who finishes where
    def reshuffling(**shares):
        res = Simulator(StubNet(sigma_pct=0.5), params(**shares), runs=300,
                        interactions=False).run(s)
        return res.positions().std(0).mean()

    assert reshuffling(common_var_share=0.95) < reshuffling(driver_var_share=0.95) / 3


def test_lap_status_follows_the_leader_not_lapped_cars():
    from strategist.sim.params import lap_status
    rows = []
    for lap in range(1, 6):
        # leader sees the SC from lap 4; a lapped car's lap 2 already overlaps it in time
        rows.append({"race_id": "r", "lap_number": lap, "position": 1,
                     "track_status": "4" if lap >= 4 else "1"})
        rows.append({"race_id": "r", "lap_number": lap, "position": 20,
                     "track_status": "4" if lap >= 2 else "1"})
    status = lap_status(pd.DataFrame(rows)).droplevel(0)
    assert status.to_dict() == {1: "GREEN", 2: "GREEN", 3: "GREEN", 4: "SC", 5: "SC"}


def test_finish_table_columns():
    res = Simulator(StubNet(), params(), runs=5).run(setup([car("A", 0.0), car("B", 0.0)], [0, 3]))
    t = res.finish_table()
    assert list(t["driver"]) == ["A", "B"] and t["p_win"].iat[0] == 1.0
    assert isinstance(t, pd.DataFrame)
