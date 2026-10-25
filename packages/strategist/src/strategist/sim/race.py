"""Lap-by-lap Monte Carlo race simulator.

Each lap, for every run and every car:

1. the lap-time network predicts mean and σ from the car's tyres, fuel, pace anchor and the
   gap to the car ahead *as produced by the simulation so far*;
2. a lap time is sampled. σ is split into a race-wide shift (drawn once per run, shared by
   every car: track conditions move the whole field and don't change the order), a race-long
   offset per driver (a driver quicker than expected stays quicker; independent noise every
   lap would average out and make outcomes far too predictable) and lap-to-lap noise;
3. pit stops add the calibrated pit loss (plus stop-to-stop variability) on the in-lap and
   reset the tyres from the next lap;
4. cars are resolved front to back. A car starting the lap < 1 s behind the car ahead
   attempts a pass with probability ``sigmoid(a_circuit + b·pace advantage + c·gap)``; if it
   fails it is held at a sampled following gap, which cascades into DRS trains;
5. safety-car laps run at a calibrated fraction of race pace and close the field up behind
   the safety car (not on the deployment lap); VSC laps keep the gaps. No overtaking under
   either.

Everything is vectorised over runs; the only Python loops are over laps and grid slots.

**Strategy variants.** ``run(setup, target=driver, variants=[...])`` simulates several
strategies for one driver at once: runs are laid out as ``variant × runs``, and every variant
reuses the same random draws for run m (common random numbers), so differences between
variants come from the strategy, not from luck. The optimiser relies on this.

Simulated outputs are counterfactual estimates (see the provenance rule in DESIGN.md).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from strategist.models.features import COMPOUNDS, GAP_CAP_S
from strategist.models.lapnet import Batch, LapTimeNet
from strategist.sim.params import SimParams

CLOSE_GAP_S = 1.0  # pairs closer than this at the start of a lap are "in a fight"
PASS_MARGIN_S = 0.2


@dataclass
class Stint:
    compound: str
    end_lap: int        # last lap on this set (the in-lap, if a stop follows)
    age_at_start: int = 0

    def __post_init__(self):
        if self.compound not in COMPOUNDS:
            raise ValueError(f"unknown compound {self.compound!r}")


@dataclass
class Strategy:
    stints: list[Stint]

    @property
    def pit_laps(self) -> list[int]:
        return [s.end_lap for s in self.stints[:-1]]

    def tyre_plan(self, total_laps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per lap 1..total_laps: compound index, tyre age (FastF1 convention: 1 on the first
        lap of a fresh set) and whether the set was fresh at fitting."""
        compound = np.zeros(total_laps + 1, dtype=np.int64)
        age = np.zeros(total_laps + 1)
        fresh = np.zeros(total_laps + 1)
        start = 1
        for s in self.stints:
            end = min(s.end_lap, total_laps) if s is not self.stints[-1] else total_laps
            laps = np.arange(start, end + 1)
            compound[laps] = COMPOUNDS.index(s.compound)
            age[laps] = s.age_at_start + (laps - start) + 1
            fresh[laps] = float(s.age_at_start == 0)
            start = end + 1
        return compound[1:], age[1:], fresh[1:]

    def describe(self) -> str:
        parts = [f"{s.compound[0]}" for s in self.stints]
        return "-".join(parts) + (f" (stops {', '.join(map(str, self.pit_laps))})"
                                  if self.pit_laps else " (no stop)")


@dataclass
class Car:
    driver: str
    q_s: float               # best qualifying lap (pace anchor)
    quali_gap_pct: float
    quali_missing: float
    grid_position: float
    strategy: Strategy
    team: str = ""


@dataclass
class RaceSetup:
    circuit: str
    total_laps: int
    track_temp: float
    cars: list[Car]
    # Start state. ``start_times``: cumulative times (s) after ``start_lap`` laps, one per car.
    # ``None`` → standing start from the grid (pre-race forecast).
    start_lap: int = 0
    start_times: np.ndarray | None = None
    # Known events (replay/backtest) or ``None`` to sample them (forecast).
    events: dict[int, str] | None = None          # lap → "SC" | "VSC" | "RED"
    retirements: dict[str, int] | None = None     # driver → last completed lap

    def driver_index(self, driver: str) -> int:
        return [c.driver for c in self.cars].index(driver)


@dataclass
class SimResult:
    drivers: list[str]
    times: np.ndarray        # [laps+1, runs, cars] cumulative time (only [1, …] if final-only)
    laps_done: np.ndarray    # [runs, cars] laps completed (retired cars stop counting)
    held_s: np.ndarray       # [runs, cars] time lost stuck behind other cars (green laps)
    start_lap: int
    variants: int = 1        # runs are laid out as variant × runs_per_variant
    events: np.ndarray | None = None  # [runs, laps+1] lap statuses actually simulated

    @property
    def runs(self) -> int:
        return self.times.shape[1]

    @property
    def runs_per_variant(self) -> int:
        return self.runs // self.variants

    def positions(self, lap: int | None = None) -> np.ndarray:
        """[runs, cars] position (1 = leader) at the end of ``lap`` (default: finish).
        Retired cars rank behind running ones, later retirements ahead of earlier ones."""
        t = self.times[-1] if lap is None else self.times[lap]
        key = np.where(np.isfinite(t), t, 1e12 - self.laps_done * 1e6)
        return key.argsort(1).argsort(1) + 1

    def by_variant(self, values: np.ndarray) -> np.ndarray:
        """Reshape a per-run array [runs, ...] to [variants, runs_per_variant, ...]."""
        return values.reshape(self.variants, self.runs_per_variant, *values.shape[1:])

    def finish_table(self) -> pd.DataFrame:
        pos = self.positions()
        final = self.times[-1]
        finite = np.isfinite(final)
        leader = np.where(finite, final, np.inf).min(axis=1, keepdims=True)
        gap = np.where(finite, final - leader, np.nan)
        median_gap = np.full(gap.shape[1], np.nan)
        some = finite.any(0)  # cars that retire in every run have no gap
        median_gap[some] = np.nanmedian(gap[:, some], 0)
        return pd.DataFrame({
            "driver": self.drivers,
            "mean_pos": pos.mean(0),
            "median_pos": np.median(pos, 0),
            "p10_pos": np.quantile(pos, 0.1, axis=0),
            "p90_pos": np.quantile(pos, 0.9, axis=0),
            "p_win": (pos == 1).mean(0),
            "p_podium": (pos <= 3).mean(0),
            "p_points": (pos <= 10).mean(0),
            "median_gap_s": median_gap,
            "mean_held_s": self.held_s.mean(0),
        }).sort_values("mean_pos").reset_index(drop=True)


class Simulator:
    def __init__(self, net: LapTimeNet, params: SimParams, runs: int = 500, seed: int = 0,
                 interactions: bool = True, record_laps: bool = True, members: int | None = None):
        """``runs`` is per strategy variant. ``record_laps=False`` keeps only the finish (much
        less memory when screening hundreds of variants); ``members`` limits the lap-time
        ensemble (faster, slightly noisier screening)."""
        self.net, self.p, self.runs, self.seed = net, params, runs, seed
        self.interactions, self.record_laps, self.members = interactions, record_laps, members

    # ---- events -----------------------------------------------------------------------------
    def _sample_events(self, rng, setup: RaceSetup, n: int) -> np.ndarray:
        status = np.full((n, setup.total_laps + 1), "GREEN", dtype=object)
        h_sc = self.p.sc_hazard.get(setup.circuit, self.p.sc_hazard_global)
        h_vsc = self.p.vsc_hazard_global
        sc_d = np.array(self.p.sc_durations or [4])
        vsc_d = np.array(self.p.vsc_durations or [2])
        for r in range(n):
            lap = max(setup.start_lap + 1, 1)
            while lap <= setup.total_laps:
                u = rng.random()
                if u < h_sc:
                    d = int(rng.choice(sc_d))
                    status[r, lap:lap + d] = "SC"
                    lap += d + 1
                elif u < h_sc + h_vsc:
                    d = int(rng.choice(vsc_d))
                    status[r, lap:lap + d] = "VSC"
                    lap += d + 1
                else:
                    lap += 1
        return status

    # ---- main loop --------------------------------------------------------------------------
    def run(self, setup: RaceSetup, target: str | None = None,
            variants: list[Strategy] | None = None) -> SimResult:
        # retired cars carry inf times; inf − inf gaps are expected and masked out
        with np.errstate(invalid="ignore"):
            return self._run(setup, target, variants)

    def _run(self, setup: RaceSetup, target: str | None,
             variants: list[Strategy] | None) -> SimResult:
        rng = np.random.default_rng(self.seed)
        M, C, L = self.runs, len(setup.cars), setup.total_laps
        K = len(variants) if variants else 1
        R = K * M
        rr = np.arange(R)
        p = self.p

        def shared(draw: np.ndarray) -> np.ndarray:
            """Common random numbers: the same draw for run m of every variant."""
            return np.tile(draw, (K,) + (1,) * (draw.ndim - 1))

        cars = pd.DataFrame([{"circuit": setup.circuit, "q_s": c.q_s,
                              "quali_gap_pct": c.quali_gap_pct, "quali_missing": c.quali_missing,
                              "grid_position": c.grid_position, "track_temp": setup.track_temp,
                              "total_laps": float(L), "compound": "MEDIUM"} for c in setup.cars])
        circuit_idx, context = self.net.static_inputs(cars)
        q = np.array([c.q_s for c in setup.cars])

        # tyre plans [R, C, L] and pit flags [R, C, L+1]; only the target varies across runs
        plans = [c.strategy.tyre_plan(L) for c in setup.cars]
        compound = np.broadcast_to(np.stack([pl[0] for pl in plans]), (R, C, L)).copy()
        age = np.broadcast_to(np.stack([pl[1] for pl in plans]), (R, C, L)).copy()
        fresh = np.broadcast_to(np.stack([pl[2] for pl in plans]), (R, C, L)).copy()
        pit_lap = np.zeros((R, C, L + 1), dtype=bool)
        for i, c in enumerate(setup.cars):
            for lap in c.strategy.pit_laps:
                if 1 <= lap < L:
                    pit_lap[:, i, lap] = True
        if variants:
            ti = setup.driver_index(target)
            pit_lap[:, ti, :] = False
            for k, strategy in enumerate(variants):
                block = slice(k * M, (k + 1) * M)
                vc, va, vf = strategy.tyre_plan(L)
                compound[block, ti], age[block, ti], fresh[block, ti] = vc, va, vf
                for lap in strategy.pit_laps:
                    if 1 <= lap < L:
                        pit_lap[block, ti, lap] = True

        # pace randomness: race-wide shift per run + race-long offset per (run, car) + lap noise
        common = float(np.clip(p.common_var_share, 0.0, 0.95))
        driver = float(np.clip(p.driver_var_share, 0.0, 0.95 - common))
        lap_share = 1.0 - common - driver
        offset_z = shared(np.sqrt(common) * rng.standard_normal((M, 1))
                          + np.sqrt(driver) * rng.standard_normal((M, C)))

        if setup.events is not None:
            status = np.full((R, L + 1), "GREEN", dtype=object)
            for lap, kind in setup.events.items():
                if 1 <= lap <= L:
                    status[:, lap] = kind
        else:
            status = shared(self._sample_events(rng, setup, M))

        retire = np.full((R, C), L + 1)
        if setup.retirements is not None:
            for i, c in enumerate(setup.cars):
                if c.driver in setup.retirements:
                    retire[:, i] = setup.retirements[c.driver]
        else:
            hit = shared(rng.random((M, C, L))) < p.dnf_hazard
            retire = np.minimum(retire, np.where(hit.any(2), hit.argmax(2) + 1, L + 1))

        if setup.start_times is not None:
            first_lap = setup.start_lap + 1
            current = np.broadcast_to(setup.start_times, (R, C)).astype(float).copy()
        else:  # standing start: lap 1 from the grid
            ratio = p.lap1_ratio.get(setup.circuit, p.lap1_ratio_global)
            grid = np.array([c.grid_position if c.grid_position > 0 else C for c in setup.cars])
            current = (q.min() * ratio + p.grid_spacing_s * (grid - 1)
                       + shared(rng.normal(0, p.start_noise_s, (M, C))))
            first_lap = 2
        current = np.where(retire < first_lap - 1, np.inf, current)
        times = np.full((L + 1 if self.record_laps else 1, R, C), np.inf)
        if self.record_laps:
            times[first_lap - 1] = current
        laps_done = np.where(np.isfinite(current), first_lap - 1, 0).astype(float)
        held_s = np.zeros((R, C))
        sc_lap_no = np.zeros(R, dtype=int)  # laps into the current safety-car period, per run

        for lap in range(first_lap, L + 1):
            prev = current
            running = np.isfinite(prev) & (retire >= lap)
            order = np.argsort(np.where(running, prev, np.inf), axis=1)
            sorted_prev = np.take_along_axis(prev, order, 1)
            gap_sorted = np.diff(sorted_prev, axis=1, prepend=np.nan)
            gap = np.empty_like(prev)
            np.put_along_axis(gap, order, gap_sorted, 1)
            leader = np.zeros((R, C), dtype=bool)
            leader[rr, order[:, 0]] = True
            gap_feat = np.where(leader | ~np.isfinite(gap), GAP_CAP_S, np.minimum(gap, GAP_CAP_S))

            laps_remaining = float(L - lap)
            b = Batch(
                circuit=circuit_idx.repeat(R),
                compound=torch.from_numpy(compound[:, :, lap - 1].reshape(-1)),
                context=context.repeat(R, 1),
                tyre_life=torch.from_numpy(age[:, :, lap - 1].reshape(-1)).float(),
                laps_remaining=torch.full((R * C,), laps_remaining),
                fresh=torch.from_numpy(fresh[:, :, lap - 1].reshape(-1)).float(),
                gap=torch.from_numpy(gap_feat.reshape(-1)).float(),
                leader=torch.from_numpy(leader.reshape(-1).astype(np.float32)),
                fuel_frac=torch.full((R * C,), laps_remaining / L),
            )
            mu_pct, sig_pct = (self.net.predict_batch(b, self.members) if self.members
                               else self.net.predict_batch(b))
            mu = (1.0 + mu_pct.numpy().reshape(R, C) / 100.0) * q
            sigma = sig_pct.numpy().reshape(R, C) / 100.0 * q
            noise = shared(rng.standard_normal((M, C)))
            lap_time = mu + sigma * (offset_z + np.sqrt(lap_share) * noise)

            kind = status[:, lap]
            sc = (kind == "SC") | (kind == "RED")
            vsc = kind == "VSC"
            green = ~(sc | vsc)
            lap_time = np.where(vsc[:, None], p.vsc_lap_ratio * mu, lap_time)

            pits = pit_lap[:, :, lap]
            loss = p.pit(setup.circuit) + p.pit_loss_sd * shared(rng.standard_normal((M, C)))
            loss = np.where(green[:, None], loss, loss * p.pit_loss_neutral_ratio)
            loss = np.where(pits, np.maximum(loss, 0.0), 0.0)

            t = prev + lap_time
            # safety car: leader at SC pace, everyone else closes up into a queue. On the lap
            # it is deployed the field hardly bunches (cars slow at different points of the lap)
            sc_lap_no = np.where(sc, sc_lap_no + 1, 0)
            if sc.any():
                factor = np.where(sc_lap_no == 1, p.sc_first_lap_factor, p.sc_compress)
                t_sc = sorted_prev.copy()
                lead_mu = np.take_along_axis(mu, order[:, :1], 1)[:, 0]
                t_sc[:, 0] = t_sc[:, 0] + p.sc_lap_ratio * lead_mu
                for k in range(1, C):
                    old_gap = sorted_prev[:, k] - sorted_prev[:, k - 1]
                    new_gap = np.maximum(p.sc_queue_gap, old_gap * factor)
                    t_sc[:, k] = t_sc[:, k - 1] + np.where(np.isfinite(old_gap), new_gap, np.inf)
                unsorted = np.empty_like(t_sc)
                np.put_along_axis(unsorted, order, t_sc, 1)
                t = np.where(sc[:, None], unsorted, t)
            t = t + loss

            if self.interactions:
                a = p.overtake_a(setup.circuit)
                pass_u = shared(rng.random((M, C)))
                hold_u = shared(rng.uniform(p.held_gap_low, p.held_gap_high, (M, C)))
                for k in range(1, C):
                    i, j = order[:, k], order[:, k - 1]
                    ti_, tj = t[rr, i], t[rr, j]
                    ok = np.isfinite(ti_) & np.isfinite(tj) & running[rr, i] & running[rr, j]
                    either_pits = pits[rr, i] | pits[rr, j]
                    start_gap = np.nan_to_num(gap[rr, i], nan=GAP_CAP_S, posinf=GAP_CAP_S,
                                              neginf=GAP_CAP_S)
                    close = ok & green & ~either_pits & (start_gap < CLOSE_GAP_S)
                    advantage = np.nan_to_num(mu[rr, j] - mu[rr, i], nan=0.0, posinf=0.0, neginf=0.0)
                    logit = a + p.overtake_pace * advantage + p.overtake_gap * start_gap
                    prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
                    passed = close & (pass_u[:, k] < prob)
                    held = close & ~passed
                    new = np.where(passed & (ti_ >= tj), tj - PASS_MARGIN_S, ti_)
                    new = np.where(held, np.maximum(new, tj + hold_u[:, k]), new)
                    # no overtaking under VSC/SC (except via the pit lane)
                    neutral = ok & ~green & ~either_pits
                    new = np.where(neutral, np.maximum(new, tj + PASS_MARGIN_S), new)
                    held_s[rr, i] += np.where(held & np.isfinite(new), new - ti_, 0.0)
                    t[rr, i] = new

            current = np.where(running & (retire >= lap), t, np.inf)
            laps_done += np.isfinite(current)
            if self.record_laps:
                times[lap] = current
        if not self.record_laps:
            times[0] = current
        return SimResult([c.driver for c in setup.cars], times, laps_done, held_s,
                         setup.start_lap, K, status)
