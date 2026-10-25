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
   the safety car; VSC laps keep the gaps. No overtaking under either.

Everything is vectorised over runs; the only Python loops are over laps and grid slots.

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


@dataclass
class SimResult:
    drivers: list[str]
    times: np.ndarray        # [laps+1, runs, cars] cumulative time; inf once retired
    start_lap: int
    events: list[np.ndarray] = field(default_factory=list)  # per run: lap statuses (forecast)

    @property
    def runs(self) -> int:
        return self.times.shape[1]

    def positions(self, lap: int | None = None) -> np.ndarray:
        """[runs, cars] position (1 = leader) at the end of ``lap`` (default: finish).
        Retired cars rank behind running ones, later retirements ahead of earlier ones."""
        lap = self.times.shape[0] - 1 if lap is None else lap
        t = self.times[lap]
        laps_done = np.isfinite(self.times).sum(0)  # [runs, cars]
        key = np.where(np.isfinite(t), t, 1e12 - laps_done * 1e6)
        return key.argsort(1).argsort(1) + 1

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
            "median_gap_s": median_gap,
        }).sort_values("mean_pos").reset_index(drop=True)


class Simulator:
    def __init__(self, net: LapTimeNet, params: SimParams, runs: int = 500, seed: int = 0,
                 interactions: bool = True):
        self.net, self.p, self.runs, self.seed = net, params, runs, seed
        self.interactions = interactions

    # ---- events -----------------------------------------------------------------------------
    def _sample_events(self, rng, setup: RaceSetup) -> np.ndarray:
        status = np.full((self.runs, setup.total_laps + 1), "GREEN", dtype=object)
        h_sc = self.p.sc_hazard.get(setup.circuit, self.p.sc_hazard_global)
        h_vsc = self.p.vsc_hazard_global
        sc_d = np.array(self.p.sc_durations or [4])
        vsc_d = np.array(self.p.vsc_durations or [2])
        for r in range(self.runs):
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
    def run(self, setup: RaceSetup) -> SimResult:
        # retired cars carry inf times; inf − inf gaps are expected and masked out
        with np.errstate(invalid="ignore"):
            return self._run(setup)

    def _run(self, setup: RaceSetup) -> SimResult:
        rng = np.random.default_rng(self.seed)
        R, C, L = self.runs, len(setup.cars), setup.total_laps
        rr = np.arange(R)
        p = self.p
        cars = pd.DataFrame([{"circuit": setup.circuit, "q_s": c.q_s,
                              "quali_gap_pct": c.quali_gap_pct, "quali_missing": c.quali_missing,
                              "grid_position": c.grid_position, "track_temp": setup.track_temp,
                              "total_laps": float(L), "compound": "MEDIUM"} for c in setup.cars])
        circuit_idx, context = self.net.static_inputs(cars)
        q = np.array([c.q_s for c in setup.cars])
        plans = [c.strategy.tyre_plan(L) for c in setup.cars]
        compound = np.stack([pl[0] for pl in plans])   # [C, L]
        age = np.stack([pl[1] for pl in plans])
        fresh = np.stack([pl[2] for pl in plans])
        pit_lap = np.zeros((C, L + 1), dtype=bool)
        for i, c in enumerate(setup.cars):
            for lap in c.strategy.pit_laps:
                if 1 <= lap < L:
                    pit_lap[i, lap] = True

        # pace randomness: race-wide shift per run + race-long offset per (run, car) + lap noise
        common = float(np.clip(p.common_var_share, 0.0, 0.95))
        driver = float(np.clip(p.driver_var_share, 0.0, 0.95 - common))
        lap_share = 1.0 - common - driver
        offset_z = (np.sqrt(common) * rng.standard_normal((R, 1))
                    + np.sqrt(driver) * rng.standard_normal((R, C)))

        if setup.events is not None:
            status = np.full((R, L + 1), "GREEN", dtype=object)
            for lap, kind in setup.events.items():
                if 1 <= lap <= L:
                    status[:, lap] = kind
        else:
            status = self._sample_events(rng, setup)

        retire = np.full((R, C), L + 1)
        if setup.retirements is not None:
            for i, c in enumerate(setup.cars):
                if c.driver in setup.retirements:
                    retire[:, i] = setup.retirements[c.driver]
        else:
            draws = rng.random((R, C, L))
            hit = draws < p.dnf_hazard
            first = np.where(hit.any(2), hit.argmax(2) + 1, L + 1)
            retire = np.minimum(retire, first)

        times = np.full((L + 1, R, C), np.inf)
        if setup.start_times is not None:
            times[setup.start_lap] = np.broadcast_to(setup.start_times, (R, C))
            first_lap = setup.start_lap + 1
        else:  # standing start: lap 1 from the grid
            ratio = p.lap1_ratio.get(setup.circuit, p.lap1_ratio_global)
            grid = np.array([c.grid_position if c.grid_position > 0 else C for c in setup.cars])
            pole = q.min()
            times[0] = 0.0
            times[1] = (pole * ratio + p.grid_spacing_s * (grid - 1)
                        + rng.normal(0, p.start_noise_s, (R, C)))
            first_lap = 2
        for lap in range(first_lap - 1, L + 1):
            times[lap] = np.where(retire < lap, np.inf, times[lap])
        sc_lap_no = np.zeros(R, dtype=int)  # laps into the current safety-car period, per run

        for lap in range(first_lap, L + 1):
            prev = times[lap - 1]
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
                compound=torch.from_numpy(np.tile(compound[:, lap - 1], R)),
                context=context.repeat(R, 1),
                tyre_life=torch.from_numpy(np.tile(age[:, lap - 1], R)).float(),
                laps_remaining=torch.full((R * C,), laps_remaining),
                fresh=torch.from_numpy(np.tile(fresh[:, lap - 1], R)).float(),
                gap=torch.from_numpy(gap_feat.reshape(-1)).float(),
                leader=torch.from_numpy(leader.reshape(-1).astype(np.float32)),
                fuel_frac=torch.full((R * C,), laps_remaining / L),
            )
            mu_pct, sig_pct = self.net.predict_batch(b)
            mu = (1.0 + mu_pct.numpy().reshape(R, C) / 100.0) * q
            sigma = sig_pct.numpy().reshape(R, C) / 100.0 * q
            lap_time = mu + sigma * (offset_z + np.sqrt(lap_share) * rng.standard_normal((R, C)))

            kind = status[:, lap]
            sc = (kind == "SC") | (kind == "RED")
            vsc = kind == "VSC"
            green = ~(sc | vsc)
            lap_time = np.where(vsc[:, None], p.vsc_lap_ratio * mu, lap_time)

            pits = np.broadcast_to(pit_lap[:, lap], (R, C))
            loss = p.pit(setup.circuit) + p.pit_loss_sd * rng.standard_normal((R, C))
            loss = np.where(green[:, None], loss, loss * p.pit_loss_neutral_ratio)
            loss = np.where(pits, np.maximum(loss, 0.0), 0.0)

            t = prev + lap_time
            # safety car: leader at SC pace, everyone else closes up into a queue. On the lap
            # it is deployed the field hardly bunches (cars slow at different points of the lap)
            sc_lap_no = np.where(sc, sc_lap_no + 1, 0)
            if sc.any():
                factor = np.where(sc_lap_no == 1, p.sc_first_lap_factor, p.sc_compress)[:, None]
                t_sc = np.take_along_axis(prev, order, 1).copy()
                lead_mu = np.take_along_axis(mu, order[:, :1], 1)[:, 0]
                t_sc[:, 0] = t_sc[:, 0] + p.sc_lap_ratio * lead_mu
                for k in range(1, C):
                    old_gap = sorted_prev[:, k] - sorted_prev[:, k - 1]
                    new_gap = np.maximum(p.sc_queue_gap, old_gap * factor[:, 0])
                    t_sc[:, k] = t_sc[:, k - 1] + np.where(np.isfinite(old_gap), new_gap, np.inf)
                unsorted = np.empty_like(t_sc)
                np.put_along_axis(unsorted, order, t_sc, 1)
                t = np.where(sc[:, None], unsorted, t)
            t = t + loss

            if self.interactions:
                a = p.overtake_a(setup.circuit)
                for k in range(1, C):
                    i, j = order[:, k], order[:, k - 1]
                    ti, tj = t[rr, i], t[rr, j]
                    ok = np.isfinite(ti) & np.isfinite(tj) & running[rr, i] & running[rr, j]
                    either_pits = pits[rr, i] | pits[rr, j]
                    start_gap = np.nan_to_num(gap[rr, i], nan=GAP_CAP_S, posinf=GAP_CAP_S,
                                              neginf=GAP_CAP_S)
                    close = ok & green & ~either_pits & (start_gap < CLOSE_GAP_S)
                    advantage = np.nan_to_num(mu[rr, j] - mu[rr, i], nan=0.0, posinf=0.0, neginf=0.0)
                    logit = a + p.overtake_pace * advantage + p.overtake_gap * start_gap
                    prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
                    passed = close & (rng.random(R) < prob)
                    held = close & ~passed
                    ti = np.where(passed & (ti >= tj), tj - PASS_MARGIN_S, ti)
                    hold_gap = rng.uniform(p.held_gap_low, p.held_gap_high, R)
                    ti = np.where(held, np.maximum(ti, tj + hold_gap), ti)
                    # no overtaking under VSC/SC (except via the pit lane)
                    neutral = ok & ~green & ~either_pits
                    ti = np.where(neutral, np.maximum(ti, tj + PASS_MARGIN_S), ti)
                    t[rr, i] = ti

            times[lap] = np.where(running & (retire >= lap), t, np.inf)

        events = [status[r] for r in range(R)] if setup.events is None else []
        return SimResult([c.driver for c in setup.cars], times, setup.start_lap, events)
