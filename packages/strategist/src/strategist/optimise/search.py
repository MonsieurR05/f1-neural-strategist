"""Strategy optimiser: enumerate legal strategies for one driver, rank them by simulation.

1. **Enumerate** every strategy from the current lap: 0..``max_stops`` further stops on a
   pit-lap grid, any compounds, subject to the dry-race rule (at least two dry compounds over
   the race), a minimum stint length and a tyre-age cap (observed maximum at the circuit
   × ``age_margin``; stints beyond the observed maximum are kept but flagged as
   extrapolation).
2. **Screen** all candidates with a few runs each. Candidates share random numbers, so even
   small run counts rank them reliably against each other.
3. **Refine** the best few, their ±1-lap neighbours and the as-raced strategy with many runs,
   and compare each to the as-raced strategy run by run.
4. **Explain** each strategy with a time budget: pit stops, tyre wear, compound pace (from the
   lap-time network's additive terms, in free air) and time lost in traffic (from the
   simulation), plus a sentence on the biggest differences to the best strategy.

The objective is expected finishing position, ties broken by expected race time.
"""

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategist.models.features import COMPOUNDS, GAP_CAP_S
from strategist.models.lapnet import LapTimeNet
from strategist.sim.params import SimParams
from strategist.sim.race import RaceSetup, Simulator, Stint, Strategy

MAX_RUNS_PER_CALL = 4000  # variants × runs per simulator call (memory bound)


@dataclass
class SearchConfig:
    max_stops: int = 2
    pit_step: int = 2
    min_stint: int = 6
    age_margin: float = 1.15
    screen_runs: int = 16
    screen_members: int | None = 2   # lap-time ensemble members used while screening
    refine_runs: int = 300
    refine_top: int = 8
    neighbours_of: int = 3
    seed: int = 0


@dataclass
class OptimiseResult:
    driver: str
    start_lap: int
    candidates: int
    table: pd.DataFrame           # refined strategies, best first
    as_raced: str
    notes: list[str] = field(default_factory=list)


def tyre_limits(laps: pd.DataFrame, circuit: str) -> dict[str, int]:
    """Maximum observed tyre age per compound at the circuit (global maximum if unseen)."""
    clean = laps[laps["exclude_reason"].isna()].dropna(subset=["tyre_life"])
    here = clean[clean["circuit"] == circuit].groupby("compound")["tyre_life"].max()
    overall = clean.groupby("compound")["tyre_life"].quantile(0.99)
    return {c: int(here.get(c, overall.get(c, 40))) for c in COMPOUNDS}


def split_at(strategy: Strategy, start_lap: int) -> tuple[list[Stint], Stint, int]:
    """Completed stints, the stint in progress after ``start_lap`` laps, and its first lap."""
    past = [s for s in strategy.stints[:-1] if s.end_lap <= start_lap]
    current = strategy.stints[len(past)]
    first = past[-1].end_lap + 1 if past else 1
    return past, current, first


def candidate_strategies(as_raced: Strategy, start_lap: int, total_laps: int,
                         limits: dict[str, int], cfg: SearchConfig) -> list[Strategy]:
    past, current, first = split_at(as_raced, start_lap)
    used = {s.compound for s in past} | {current.compound}
    cap = {c: int(limits[c] * cfg.age_margin) for c in COMPOUNDS}
    grid = list(range(start_lap + 1, total_laps - cfg.min_stint + 1, cfg.pit_step))
    out = []
    for n in range(cfg.max_stops + 1):
        for compounds in itertools.product(COMPOUNDS, repeat=n):
            if len(used | set(compounds)) < 2:
                continue
            for pits in itertools.combinations(grid, n):
                ends = [*pits, total_laps]
                ages = [current.age_at_start + ends[0] - first + 1]
                ages += [ends[k + 1] - ends[k] for k in range(n)]
                lengths = [ends[0] - first + 1] + ages[1:]
                if any(length < cfg.min_stint for length in lengths[1:]):
                    continue
                if any(a > cap[c] for a, c in zip(ages, [current.compound, *compounds], strict=True)):
                    continue
                stints = [*past, Stint(current.compound, ends[0], current.age_at_start)]
                stints += [Stint(c, e, 0) for c, e in zip(compounds, ends[1:], strict=True)]
                out.append(Strategy(stints))
    return out


def _key(s: Strategy) -> tuple:
    return tuple((st.compound, st.end_lap, st.age_at_start) for st in s.stints[:-1]) + (
        s.stints[-1].compound,)


def neighbours(strategy: Strategy, start_lap: int, total_laps: int, cfg: SearchConfig) -> list[Strategy]:
    """Each future stop moved one lap earlier or later."""
    out = []
    for i, st in enumerate(strategy.stints[:-1]):
        if st.end_lap <= start_lap:
            continue
        for d in (-1, 1):
            ends = [s.end_lap for s in strategy.stints[:-1]]
            ends[i] += d
            future = [e for e in ends if e > start_lap]
            bounds = [start_lap, *future, total_laps]
            if ends[i] <= start_lap or any(b - a < 1 for a, b in zip(bounds, bounds[1:], strict=False)):
                continue
            stints = [Stint(s.compound, e, s.age_at_start) for s, e in zip(strategy.stints, ends, strict=False)]
            out.append(Strategy([*stints, strategy.stints[-1]]))
    return out


def simulate_variants(net: LapTimeNet, params: SimParams, setup: RaceSetup, driver: str,
                      strategies: list[Strategy], runs: int, seed: int,
                      members: int | None = None) -> dict[str, np.ndarray]:
    """Per variant × run: the driver's finishing position, race time and time held up."""
    ti = setup.driver_index(driver)
    chunk = max(1, MAX_RUNS_PER_CALL // runs)
    pos, time, held = [], [], []
    for start in range(0, len(strategies), chunk):
        batch = strategies[start:start + chunk]
        res = Simulator(net, params, runs=runs, seed=seed, record_laps=False,
                        members=members).run(setup, target=driver, variants=batch)
        pos.append(res.by_variant(res.positions()[:, ti]))
        time.append(res.by_variant(res.times[-1][:, ti]))
        held.append(res.by_variant(res.held_s[:, ti]))
    return {"pos": np.concatenate(pos), "time": np.concatenate(time), "held": np.concatenate(held)}


def summarise(strategies: list[Strategy], sims: dict[str, np.ndarray],
              reference: int | None = None) -> pd.DataFrame:
    pos, time = sims["pos"], sims["time"]
    finite = np.isfinite(time)
    rows = []
    for k, s in enumerate(strategies):
        row = {
            "strategy": s.describe(), "stops": len(s.pit_laps),
            "expected_pos": pos[k].mean(), "p10_pos": np.quantile(pos[k], 0.1),
            "p90_pos": np.quantile(pos[k], 0.9), "p_win": (pos[k] == 1).mean(),
            "p_podium": (pos[k] <= 3).mean(), "p_points": (pos[k] <= 10).mean(),
            "mean_time_s": float(time[k][finite[k]].mean()) if finite[k].any() else np.nan,
            "traffic_s": float(sims["held"][k].mean()),
        }
        if reference is not None:
            both = finite[k] & finite[reference]
            row["p_ahead_of_as_raced"] = float(((pos[k] < pos[reference]) + 0.5 * (pos[k] == pos[reference])).mean())
            row["time_vs_as_raced_s"] = float((time[k][both] - time[reference][both]).mean()) if both.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def time_budget(net: LapTimeNet, params: SimParams, setup: RaceSetup, driver: str,
                strategy: Strategy) -> dict[str, float]:
    """Free-air time budget over the remaining laps: pit stops, tyre wear, compound pace."""
    car = setup.cars[setup.driver_index(driver)]
    L, start = setup.total_laps, setup.start_lap
    comp, age, fresh = strategy.tyre_plan(L)
    laps = np.arange(start + 1, L + 1)
    rows = pd.DataFrame({
        "circuit": setup.circuit, "q_s": car.q_s, "quali_gap_pct": car.quali_gap_pct,
        "quali_missing": car.quali_missing, "grid_position": car.grid_position,
        "track_temp": setup.track_temp, "total_laps": float(L),
        "compound": [COMPOUNDS[c] for c in comp[laps - 1]], "tyre_life": age[laps - 1],
        "fresh_tyre": fresh[laps - 1], "laps_remaining": (L - laps).astype(float),
        "fuel_frac": (L - laps) / L, "gap_ahead_s": GAP_CAP_S, "is_leader": 0.0,
    })
    parts = net.components(rows)
    stops = [p for p in strategy.pit_laps if p > start]
    return {
        "pit_stops_s": len(stops) * params.pit(setup.circuit),
        "tyre_wear_s": float(parts["deg"].sum()),
        "compound_pace_s": float((parts["offset"] + parts["conditions"]).sum()),
    }


def explain(row: pd.Series, best: pd.Series) -> str:
    if row["strategy"] == best["strategy"]:
        return "Best expected finishing position."
    labels = {"pit_stops_s": "in the pits", "tyre_wear_s": "to tyre wear",
              "compound_pace_s": "from compound pace", "traffic_s": "in traffic"}
    diffs = {k: row[k] - best[k] for k in labels}
    top = sorted(diffs.items(), key=lambda kv: -abs(kv[1]))[:3]
    parts = [f"{'+' if d > 0 else '−'}{abs(d):.1f} s {labels[k]}" for k, d in top if abs(d) >= 0.5]
    gap = row["expected_pos"] - best["expected_pos"]
    return (f"vs best: {', '.join(parts) or 'similar time budget'}; "
            f"{gap:+.2f} positions on average.")


def optimise(net: LapTimeNet, params: SimParams, setup: RaceSetup, driver: str,
             limits: dict[str, int], cfg: SearchConfig | None = None) -> OptimiseResult:
    cfg = cfg or SearchConfig()
    car = setup.cars[setup.driver_index(driver)]
    as_raced = car.strategy
    candidates = candidate_strategies(as_raced, setup.start_lap, setup.total_laps, limits, cfg)
    notes = []
    if not candidates:
        candidates = [as_raced]
        notes.append("No legal alternative strategies from this lap; showing the as-raced one.")

    screen = summarise(candidates, simulate_variants(net, params, setup, driver, candidates,
                                                     cfg.screen_runs, cfg.seed, cfg.screen_members))
    order = screen.sort_values(["expected_pos", "mean_time_s"]).index
    shortlist = [candidates[i] for i in order[:cfg.refine_top]]
    for s in shortlist[:cfg.neighbours_of]:
        shortlist += neighbours(s, setup.start_lap, setup.total_laps, cfg)
    shortlist.append(as_raced)
    unique, seen = [], set()
    for s in shortlist:
        if _key(s) not in seen:
            seen.add(_key(s))
            unique.append(s)
    ref = next(i for i, s in enumerate(unique) if _key(s) == _key(as_raced))

    table = summarise(unique, simulate_variants(net, params, setup, driver, unique,
                                                cfg.refine_runs, cfg.seed + 1), reference=ref)
    budgets = pd.DataFrame([time_budget(net, params, setup, driver, s) for s in unique])
    table = pd.concat([table, budgets], axis=1)
    table["as_raced"] = [i == ref for i in range(len(unique))]
    table["extrapolated"] = [
        any(age > limits[st.compound]
            for st, age in remaining_stint_ages(s, setup.start_lap, setup.total_laps))
        for s in unique]
    table["stints"] = [[{"compound": st.compound, "end_lap": st.end_lap, "age_at_start": st.age_at_start}
                        for st in s.stints] for s in unique]
    table = table.sort_values(["expected_pos", "mean_time_s"]).reset_index(drop=True)
    best = table.iloc[0]
    table["explanation"] = [explain(r, best) for _, r in table.iterrows()]
    if table["extrapolated"].any():
        notes.append("Strategies marked extrapolated run tyres longer than ever observed at this "
                     "circuit; their degradation is the network's extrapolation.")
    return OptimiseResult(driver, setup.start_lap, len(candidates), table, as_raced.describe(), notes)


def remaining_stint_ages(strategy: Strategy, start_lap: int, total_laps: int):
    """(stint, tyre age at the end of the stint) for the stints still running after start_lap."""
    start = 1
    for st in strategy.stints:
        end = total_laps if st is strategy.stints[-1] else st.end_lap
        if end > start_lap:
            yield st, st.age_at_start + end - start + 1
        start = end + 1
