"""Export the lap-time network, simulator parameters and race setups as JSON for the Rust
simulator (``crates/strategist-sim``), and write the fixtures its parity tests replay.

    python -m strategist.sim.export --config configs/api.yaml
    python -m strategist.sim.export --web apps/web/public/sim   # what the browser loads

``--web`` writes the model, the parameters and the replay setup (after lap 1) of every race
the app can simulate, for the WebAssembly build the web app runs.

This simulator stays the reference: every published number comes from it. The Rust port must
reproduce it, so each fixture holds a setup, the random numbers the Python run drew, and what
it produced. Rust replays the same draws and has to land on the same lap times. The draws are
recorded by a stand-in for the NumPy generator that builds every distribution from three
primitives (standard normal, uniform on [0, 1), a choice from a list), which is all the Rust
side has to implement.

Fixtures are generated, not committed (CI regenerates them before ``cargo test``), so the
port is always checked against the current Python code and models.
"""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch import nn

from strategist.config import load_config
from strategist.models.features import COMPOUNDS, prepare
from strategist.models.lapnet import CONTEXT_FEATURES, Batch, LapTimeNet
from strategist.sim.history import setup_from_history
from strategist.sim.params import SimParams
from strategist.sim.race import RaceSetup, SimResult, Simulator, Stint, Strategy

log = logging.getLogger(__name__)


# ---- model, parameters and setups as plain data ---------------------------------------------
def _layers(module: nn.Module) -> list[dict]:
    linear = [module] if isinstance(module, nn.Linear) else [m for m in module if isinstance(m, nn.Linear)]
    return [{"w": m.weight.tolist(), "b": m.bias.tolist()} for m in linear]


def model_dict(net: LapTimeNet) -> dict:
    """Weights of every ensemble member plus the input standardisation and circuit index."""
    members = []
    for m in net.modules:
        members.append({
            "circuit_emb": m.circuit_emb.weight.tolist(),
            "compound_emb": m.compound_emb.weight.tolist(),
            "tau": m.tau,
            **{name: _layers(getattr(m, name))
               for name in ("base", "conditions", "fuel", "offset", "deg", "traffic", "sigma")},
            "interaction": _layers(m.interaction) if m.interaction is not None else None,
        })
    return {
        "circuit_index": {str(k): int(v) for k, v in net.circuit_index.items()},
        "context_mean": [float(net.mean[c]) for c in CONTEXT_FEATURES],
        "context_std": [float(net.std[c]) for c in CONTEXT_FEATURES],
        "members": members,
    }


def strategy_list(strategy: Strategy) -> list[dict]:
    return [{"compound": s.compound, "end_lap": int(s.end_lap), "age_at_start": int(s.age_at_start)}
            for s in strategy.stints]


def setup_dict(setup: RaceSetup) -> dict:
    """A race setup without the in-race update (``pace_paths`` is not ported)."""
    if setup.pace_paths is not None:
        raise ValueError("pace_paths are not supported by the Rust simulator")
    return {
        "circuit": setup.circuit, "total_laps": int(setup.total_laps),
        "track_temp": float(setup.track_temp),
        "cars": [{"driver": c.driver, "team": c.team, "q_s": float(c.q_s),
                  "quali_gap_pct": float(c.quali_gap_pct), "quali_missing": float(c.quali_missing),
                  "grid_position": float(c.grid_position), "strategy": strategy_list(c.strategy)}
                 for c in setup.cars],
        "start_lap": int(setup.start_lap),
        "start_times": None if setup.start_times is None else [float(t) for t in setup.start_times],
        "events": None if setup.events is None
        else [[int(lap), kind] for lap, kind in sorted(setup.events.items())],
        "retirements": None if setup.retirements is None
        else {d: int(lap) for d, lap in setup.retirements.items()},
        "hardness": setup.hardness,
    }


# ---- parity fixtures -------------------------------------------------------------------------
class RecordingRng:
    """Stands in for ``np.random.default_rng`` and keeps every primitive draw, in order."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.normals: list[np.ndarray] = []
        self.uniforms: list[np.ndarray] = []
        self.choices: list[int] = []

    def standard_normal(self, size=None):
        x = self.rng.standard_normal(size)
        self.normals.append(np.ravel(x))
        return x

    def random(self, size=None):
        x = self.rng.random(size)
        self.uniforms.append(np.ravel(x))
        return x

    def normal(self, loc, scale, size=None):
        return loc + scale * self.standard_normal(size)

    def uniform(self, low, high, size=None):
        return low + (high - low) * self.random(size)

    def choice(self, options):
        x = int(self.rng.choice(options))
        self.choices.append(x)
        return x

    def draws(self) -> dict:
        flat = lambda parts: np.concatenate(parts).tolist() if parts else []  # noqa: E731
        return {"normal": flat(self.normals), "uniform": flat(self.uniforms), "choice": self.choices}


def _finite(a: np.ndarray) -> list:
    """Flat list with ``None`` for retired cars' infinite times (JSON has no infinity)."""
    return [float(x) if np.isfinite(x) else None for x in np.ravel(a)]


def _summaries(res: SimResult) -> dict:
    """The finish table and per-lap bands the app shows, for a run with every lap recorded."""
    finish = res.finish_table()
    return {"finish": {col: _finite(finish[col]) if col != "driver" else finish[col].tolist()
                       for col in finish.columns},
            "bands": {d: {k: _finite(v) for k, v in b.items()} for d, b in res.lap_bands().items()}}


def simulation_case(name: str, net: LapTimeNet, params: SimParams, setup: RaceSetup, runs: int,
                    seed: int = 0, target: str | None = None,
                    variants: list[Strategy] | None = None, **options) -> dict:
    """Run the Python simulator with a recording generator; keep inputs, draws and outputs."""
    summaries = variants is None and options.get("record_laps", True)
    rec = RecordingRng(seed)
    with mock.patch("numpy.random.default_rng", lambda *_: rec):
        res: SimResult = Simulator(net, params, runs=runs, seed=seed, **options).run(
            setup, target=target, variants=variants)
    return {
        "name": name, "setup": setup_dict(setup),
        "options": {"runs": runs, "target": target,
                    "variants": None if variants is None else [strategy_list(v) for v in variants],
                    "interactions": options.get("interactions", True),
                    "record_laps": options.get("record_laps", True),
                    "members": options.get("members")},
        "draws": rec.draws(),
        "expected": {"shape": list(res.times.shape), "times": _finite(res.times),
                     "laps_done": np.ravel(res.laps_done).tolist(),
                     "held_s": np.ravel(res.held_s).tolist(),
                     "status": [[str(s) for s in row] for row in res.events],
                     **(_summaries(res) if summaries else {})},
    }


def network_case(net: LapTimeNet, rows: int = 400, seed: int = 0) -> dict:
    """Random inputs across the network's range with torch's answers (full and 2-member)."""
    rng = np.random.default_rng(seed)
    n_circuits = len(net.circuit_index) + 1
    hardness = rng.choice([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0], rows)
    leader = (rng.random(rows) < 0.1).astype(np.float32)
    gap = np.where(leader > 0, 10.0, rng.uniform(0.0, 10.0, rows))
    inputs = {
        "circuit": rng.integers(0, n_circuits, rows), "compound": rng.integers(0, len(COMPOUNDS), rows),
        "context": rng.normal(0.0, 1.0, (rows, len(CONTEXT_FEATURES))),
        "tyre_life": rng.integers(1, 46, rows).astype(float),
        "laps_remaining": rng.integers(0, 71, rows).astype(float),
        "fresh": (rng.random(rows) < 0.7).astype(float), "gap": gap, "leader": leader,
        "fuel_frac": rng.random(rows), "hardness": hardness,
    }
    batch = Batch(**{k: torch.tensor(v, dtype=torch.long if k in ("circuit", "compound")
                                     else torch.float32) for k, v in inputs.items()})
    mu, sigma = net.predict_batch(batch)
    mu2, sigma2 = net.predict_batch(batch, 2)
    as_lists = {k: getattr(batch, k).tolist() for k in inputs}  # as float32 saw them
    return {"inputs": as_lists, "mu": mu.tolist(), "sigma": sigma.tolist(),
            "mu_2": mu2.tolist(), "sigma_2": sigma2.tolist()}


def history_setup(laps: pd.DataFrame, raw_root: Path, races: pd.DataFrame, race_id: str,
                  start_lap: int) -> RaceSetup:
    r = races.loc[race_id]
    results = pd.read_parquet(raw_root / str(r["year"]) / f"{int(r['round']):02d}" / "results.parquet")
    return setup_from_history(prepare(laps[laps["race_id"] == race_id]), results, start_lap)[0]


def _edited(strategy: Strategy, pit_lap: int, compound: str) -> Strategy:
    """A one-stop alternative for the variant fixture."""
    return Strategy([Stint(strategy.stints[0].compound, pit_lap, strategy.stints[0].age_at_start),
                     Stint(compound, 999)])


def fixture_cases(net: LapTimeNet, params: SimParams, laps: pd.DataFrame, races: pd.DataFrame,
                  raw_root: Path) -> list[dict]:
    """Small runs that between them reach every branch of the simulator."""
    setup = lambda race_id, start: history_setup(laps, raw_root, races, race_id, start)  # noqa: E731
    cases = []

    replay = setup("2024_01", 1)
    cases.append(simulation_case("replay_2024_bahrain", net, params, replay, runs=6))

    # 2022 Bahrain: a late safety car, retirements, stops under the safety car
    cases.append(simulation_case("replay_2022_bahrain_safety_car", net, params,
                                 setup("2022_01", 30), runs=6))

    forecast = setup("2025_04", 1)
    forecast.start_lap, forecast.start_times = 0, None      # standing start from the grid
    forecast.events, forecast.retirements = None, None      # sampled safety cars and retirements
    cases.append(simulation_case("forecast_2025_bahrain", net, params, forecast, runs=8, seed=3))

    mid = setup("2024_01", 12)
    mid.events, mid.retirements = None, None
    driver = mid.cars[2].driver
    raced = mid.cars[2].strategy
    variants = [raced, _edited(raced, 20, "HARD"), _edited(raced, 26, "MEDIUM")]
    cases.append(simulation_case("variants_2024_bahrain", net, params, mid, runs=5, seed=1,
                                 target=driver, variants=variants, record_laps=False, members=2))

    cases.append(simulation_case("free_passing_2024_bahrain", net, params, replay, runs=4,
                                 seed=2, interactions=False))
    return cases


def web_bundle(net: LapTimeNet, params: SimParams, laps: pd.DataFrame, races: pd.DataFrame,
               raw_root: Path, out: Path) -> int:
    """Model, parameters and one replay setup per simulatable race (dry, representative
    qualifying), as the browser simulator needs them. Returns the number of races."""
    (out / "races").mkdir(parents=True, exist_ok=True)
    write = lambda path, obj: path.write_text(json.dumps(obj, separators=(",", ":")),  # noqa: E731
                                              encoding="utf-8")
    write(out / "model.json", model_dict(net))
    write(out / "params.json", asdict(params))
    simulatable = races[~races["wet"] & races["quali_representative"]].index
    for race_id in simulatable:
        write(out / "races" / f"{race_id}.json", setup_dict(history_setup(laps, raw_root, races, race_id, 1)))
    return len(simulatable)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="configs/api.yaml",
                        help="API profile naming the data config, model directory and model tag")
    parser.add_argument("--out", default="crates/strategist-sim/tests/fixtures")
    parser.add_argument("--web", help="write the browser bundle to this directory instead")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    data_cfg = load_config(cfg["data_config"])
    processed, raw_root = Path(data_cfg["paths"]["processed"]), Path(data_cfg["paths"]["raw"])
    laps = pd.read_parquet(processed / "laps.parquet")
    races = pd.read_parquet(processed / "races.parquet").set_index("race_id")
    model_dir, tag = Path(cfg["model_dir"]), cfg["model_tag"]
    torch.set_num_threads(1)
    net = LapTimeNet.load(model_dir / f"lap_net_{tag}.pt")
    params = SimParams.load(model_dir / f"sim_params_{tag}.json")

    if args.web:
        n = web_bundle(net, params, laps, races, raw_root, Path(args.web))
        log.info("browser bundle for model %s: %d races in %s", tag, n, args.web)
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write = lambda name, obj: (out / name).write_text(json.dumps(obj), encoding="utf-8")  # noqa: E731
    write("model.json", model_dict(net))
    write("params.json", asdict(params))
    write("network.json", network_case(net))
    for case in fixture_cases(net, params, laps, races, raw_root):
        write(f"sim_{case['name']}.json", case)
        log.info("%s: %d draws", case["name"], sum(len(v) for v in case["draws"].values()))
    log.info("fixtures for model %s in %s", tag, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
