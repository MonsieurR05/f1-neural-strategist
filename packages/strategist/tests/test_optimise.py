import numpy as np
import pandas as pd

from strategist.optimise.search import (
    SearchConfig,
    candidate_strategies,
    explain,
    neighbours,
    optimise,
    remaining_stint_ages,
    split_at,
)
from strategist.sim.race import Stint, Strategy
from tests.test_sim import StubNet, car, params, setup

LIMITS = {"SOFT": 20, "MEDIUM": 30, "HARD": 40}
CFG = SearchConfig(max_stops=2, pit_step=1, min_stint=5, age_margin=1.0, screen_runs=4,
                   refine_runs=8, refine_top=3, neighbours_of=1)


def test_candidates_respect_compound_rule_stint_lengths_and_age_caps():
    as_raced = Strategy([Stint("SOFT", 15), Stint("HARD", 57)])
    cands = candidate_strategies(as_raced, start_lap=1, total_laps=57, limits=LIMITS, cfg=CFG)
    assert cands
    for s in cands:
        compounds = {st.compound for st in s.stints}
        assert len(compounds) >= 2
        assert s.stints[0].compound == "SOFT"  # tyres already fitted can't change
        ends = [0, *[st.end_lap for st in s.stints[:-1]], 57]
        assert all(b - a >= 5 for a, b in zip(ends[1:], ends[2:], strict=False))
        for st, age in remaining_stint_ages(s, 1, 57):
            assert age <= LIMITS[st.compound]
    # no single-compound or zero-stop plans when only softs have been used
    assert all(len(s.stints) > 1 for s in cands)


def test_candidates_keep_completed_stints_mid_race():
    as_raced = Strategy([Stint("SOFT", 12), Stint("MEDIUM", 35), Stint("HARD", 57)])
    cands = candidate_strategies(as_raced, start_lap=20, total_laps=57, limits=LIMITS, cfg=CFG)
    past, current, first = split_at(as_raced, 20)
    assert [s.end_lap for s in past] == [12] and current.compound == "MEDIUM" and first == 13
    assert all(s.stints[0] == Stint("SOFT", 12) for s in cands)
    # mediums from lap 13 to 57 (45 laps) exceed the 30-lap cap, so a stop is still needed...
    assert all(len(s.stints) >= 3 for s in cands)
    # ...but with long-lasting mediums, staying out is legal (soft and medium both used)
    long_mediums = {**LIMITS, "MEDIUM": 50}
    cands = candidate_strategies(as_raced, start_lap=20, total_laps=57, limits=long_mediums, cfg=CFG)
    assert any(len(s.stints) == 2 for s in cands)


def test_neighbours_move_only_future_stops():
    s = Strategy([Stint("SOFT", 12), Stint("MEDIUM", 35), Stint("HARD", 57)])
    moved = neighbours(s, start_lap=20, total_laps=57, cfg=CFG)
    assert sorted(m.pit_laps for m in moved) == [[12, 34], [12, 36]]


def test_explain_mentions_largest_differences():
    best = pd.Series({"strategy": "A", "expected_pos": 3.0, "pit_stops_s": 24.0,
                      "tyre_wear_s": 40.0, "compound_pace_s": 10.0, "traffic_s": 2.0})
    other = best.copy()
    other.update({"strategy": "B", "expected_pos": 4.2, "pit_stops_s": 48.0, "tyre_wear_s": 25.0})
    text = explain(other, best)
    assert "+24.0 s in the pits" in text and "−15.0 s to tyre wear" in text and "+1.20" in text


class StubNetWithComponents(StubNet):
    def components(self, df):
        n = len(df)
        return pd.DataFrame({"deg": 0.05 * df["tyre_life"].to_numpy(), "offset": np.zeros(n),
                             "conditions": np.zeros(n)}, index=df.index)


def test_optimise_end_to_end_prefers_fewer_stops_without_degradation():
    # no tyre wear in the stub: every stop is pure time loss, so the best plan stops once
    cars = [car("A", 0.0, [Stint("SOFT", 20), Stint("HARD", 99)]),
            car("B", 0.1, [Stint("SOFT", 10), Stint("MEDIUM", 20), Stint("HARD", 99)])]
    s = setup(cars, [0.0, 2.0], laps=30)
    res = optimise(StubNetWithComponents(), params(), s, "B", LIMITS, CFG)
    best = res.table.iloc[0]
    assert best["stops"] == 1 and res.table["as_raced"].sum() == 1
    raced = res.table[res.table["as_raced"]].iloc[0]
    assert raced["stops"] == 2 and raced["time_vs_as_raced_s"] == 0.0
    assert best["time_vs_as_raced_s"] < -15.0
    assert res.candidates > 10 and "Best" in best["explanation"]
