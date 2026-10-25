"""Backtest the race simulator by replaying real races.

Usage (from the repository root):

    python -m strategist.sim.backtest --config configs/sim.yaml
    python -m strategist.sim.backtest --config configs/sim.yaml --split test --final

Each race of the configured circuit is replayed from the actual order and gaps after lap 1,
with the strategies, safety-car/VSC laps and retirements that actually happened, and N Monte
Carlo runs. That isolates what the simulator is responsible for (pace, tyres, fuel, traffic,
overtaking, pit losses) from what nobody can forecast (lap-1 incidents, crashes).

The lap-time network and simulator parameters are fitted on the training seasons only and
cached under ``model_dir``. Training-season races are reported as in-sample; races from the
evaluation split are held out. Scoring the test season requires ``--final``.

Compared against:
- **lap-1 order**: positions frozen after lap 1 (the simulator's starting point);
- **grid order**;
- **no interactions**: the same simulator with free passing (an ablation of the traffic and
  overtaking model).
"""

import argparse
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from strategist.config import load_config
from strategist.eval.run import md_table
from strategist.models.features import clean_laps, prepare
from strategist.models.lapnet import LapTimeNet
from strategist.sim.history import setup_from_history
from strategist.sim.params import SimParams, calibrate
from strategist.sim.race import Simulator

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger(__name__)


def fitted_models(cfg, data_cfg, eval_cfg, laps_raw, races):
    """lap_net and simulator parameters fitted on the training seasons (cached)."""
    train_years = data_cfg["split"]["train"]
    tag = f"{min(train_years)}-{max(train_years)}"
    model_dir = Path(cfg["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    net_path, params_path = model_dir / f"lap_net_{tag}.pt", model_dir / f"sim_params_{tag}.json"

    if net_path.exists():
        net = LapTimeNet.load(net_path)
    else:
        log.info("training lap_net on %s", train_years)
        train = prepare(clean_laps(laps_raw[laps_raw["year"].isin(train_years)]))
        net = LapTimeNet(**eval_cfg["models"]["lap_net"], seed=eval_cfg["seed"]).fit(train)
        net.save(net_path)
    if params_path.exists():
        params = SimParams.load(params_path)
    else:
        log.info("calibrating simulator parameters on %s", train_years)
        params = calibrate(laps_raw, races, net, train_years)
        params.save(params_path)
    return net, params


def score(result, outcome: pd.DataFrame) -> dict:
    table = result.finish_table().set_index("driver")
    fin = outcome[outcome["finished"]].set_index("driver")
    sim = table.loc[fin.index]
    actual = fin["actual_pos"]
    lead_lap = fin["actual_gap_s"].notna()
    by_lap = outcome.attrs["positions_by_lap"]
    lap_err = []
    for lap in by_lap.index:
        if lap <= result.start_lap or lap >= result.times.shape[0]:
            continue
        sim_pos = np.median(result.positions(lap), 0)
        act = by_lap.loc[lap].to_numpy()
        ok = ~np.isnan(act)
        lap_err.append(np.abs(sim_pos[ok] - act[ok]).mean())
    winner = actual.idxmin()
    return {
        "pos_mae": float(np.abs(sim["median_pos"] - actual).mean()),
        "spearman": float(spearmanr(sim["mean_pos"], actual).statistic),
        "coverage_80": float(((sim["p10_pos"] <= actual) & (actual <= sim["p90_pos"])).mean()),
        "p_actual_winner": float(table.loc[winner, "p_win"]),
        "gap_mae_s": float(np.abs(sim.loc[lead_lap, "median_gap_s"] - fin.loc[lead_lap, "actual_gap_s"]).mean()),
        "lap_pos_mae": float(np.mean(lap_err)),
    }


def baseline_score(order: list[str], outcome: pd.DataFrame) -> dict:
    fin = outcome[outcome["finished"]].set_index("driver")
    rank = pd.Series(range(1, len(order) + 1), index=order).loc[fin.index].rank()
    return {"pos_mae": float(np.abs(rank - fin["actual_pos"]).mean()),
            "spearman": float(spearmanr(rank, fin["actual_pos"]).statistic)}


def plot_race(result, outcome, title, path):
    table = result.finish_table().set_index("driver")
    o = outcome.set_index("driver").sort_values("actual_pos")
    drivers = o.index.tolist()
    y = np.arange(len(drivers))
    blue, orange, ink, ink2, grid = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(drivers) + 1.4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    t = table.loc[drivers]
    ax.hlines(y, t["p10_pos"], t["p90_pos"], color=blue, lw=4, alpha=0.35, label="simulated 10–90%")
    ax.plot(t["median_pos"], y, "o", color=blue, ms=6, label="simulated median")
    ax.plot(o["actual_pos"], y, "D", color=orange, ms=6, mec="#fcfcfb", mew=1.5, label="actual")
    ax.set_yticks(y, drivers, color=ink2)
    ax.invert_yaxis()
    ax.set_xlabel("finishing position", color=ink2)
    ax.set_xticks(range(1, len(drivers) + 1))
    ax.tick_params(colors=ink2)
    ax.grid(axis="x", color=grid)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.legend(loc="upper right", frameon=False)  # front-runners sit left, so this corner is empty
    ax.set_title(title, loc="left", color=ink, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="configs/sim.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--final", action="store_true", help="required to score the test season")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.split == "test" and not args.final:
        parser.error("the test season is for the final comparison; pass --final to score it")

    cfg = load_config(args.config)
    data_cfg, eval_cfg = load_config(cfg["data_config"]), load_config(cfg["eval_config"])
    processed = Path(data_cfg["paths"]["processed"])
    laps_raw = pd.read_parquet(processed / "laps.parquet")
    races = pd.read_parquet(processed / "races.parquet")
    net, params = fitted_models(cfg, data_cfg, eval_cfg, laps_raw, races)

    split = data_cfg["split"]
    years = split["train"] + split[args.split]
    targets = races[(races["circuit"] == cfg["circuit"]) & races["year"].isin(years) & ~races["wet"]]
    rows, figure = [], None
    for race in targets.sort_values("year").itertuples():
        race_laps = prepare(laps_raw[laps_raw["race_id"] == race.race_id])
        results = pd.read_parquet(Path(data_cfg["paths"]["raw"]) / str(race.year)
                                  / f"{race.round:02d}" / "results.parquet")
        setup, outcome = setup_from_history(race_laps, results, start_lap=cfg["start_lap"])
        role = "held out" if race.year in split[args.split] else "in-sample"
        log.info("%s %s (%s): %d cars, %d events", race.race_id, race.event_name, role,
                 len(setup.cars), len(setup.events))
        sim = Simulator(net, params, runs=cfg["runs"], seed=cfg["seed"]).run(setup)
        free = Simulator(net, params, runs=cfg["runs"], seed=cfg["seed"], interactions=False).run(setup)
        lap1_order = [c.driver for c in setup.cars]
        grid_order = [c.driver for c in sorted(setup.cars, key=lambda c: c.grid_position)]
        base = {"race": race.race_id, "role": role}
        rows.append({**base, "model": "simulator", **score(sim, outcome)})
        rows.append({**base, "model": "no interactions", **score(free, outcome)})
        rows.append({**base, "model": "lap-1 order", **baseline_score(lap1_order, outcome)})
        rows.append({**base, "model": "grid order", **baseline_score(grid_order, outcome)})
        if role == "held out":
            figure = Path(cfg["figure_dir"]) / f"sim_backtest_{race.race_id}.png"
            plot_race(sim, outcome, f"{race.event_name} {race.year}: simulated vs actual finish "
                      f"({cfg['runs']} runs, held out)", figure)
            held_table = sim.finish_table().merge(outcome[["driver", "actual_pos", "actual_gap_s"]],
                                                  on="driver")
            held_name = f"{race.event_name} {race.year}"

    scores = pd.DataFrame(rows)
    report = render(cfg, args.split, params, scores, figure, held_table, held_name)
    path = Path(cfg["report_dir"]) / f"sim_backtest_{args.split}.md"
    path.write_text(report, encoding="utf-8")
    log.info("wrote %s", path)
    return 0


def render(cfg, split, params, scores, figure, held_table, held_name) -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    metrics = ["pos_mae", "spearman", "coverage_80", "p_actual_winner", "gap_mae_s", "lap_pos_mae"]
    order = ["simulator", "no interactions", "lap-1 order", "grid order"]
    summary = (scores.groupby(["role", "model"])[metrics].mean().reindex(order, level=1)
               .reset_index().sort_values("role", kind="stable"))
    per_race = scores[scores["model"] == "simulator"][["race", "role", *metrics]]
    held = held_table.sort_values("actual_pos")[
        ["driver", "actual_pos", "median_pos", "p10_pos", "p90_pos", "p_win", "p_podium",
         "actual_gap_s", "median_gap_s"]]
    pit = params.pit(cfg["circuit"])
    lines = [
        f"# Race simulator backtest: {cfg['circuit']}",
        "",
        f"_Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC from commit `{commit}` by "
        "`python -m strategist.sim.backtest`. Do not edit by hand._",
        "",
        "**Setup.** Each race is replayed from the actual order and gaps after lap 1 with the "
        "strategies, safety-car/VSC laps and retirements that actually happened, "
        f"{cfg['runs']} Monte Carlo runs. Lap-time network and simulator parameters are fitted on "
        "the training seasons only; training-season races are in-sample, the evaluation season "
        f"({split}) is held out. Actual order is road order at the line (official results include "
        "post-race penalties).",
        "",
        "**Metrics.** `pos_mae`: mean |median simulated − actual| finishing position. `spearman`: "
        "rank correlation of mean simulated and actual finishing order. `coverage_80`: share of "
        "drivers whose actual position falls in the simulated 10–90% range (≈0.8 if calibrated). "
        "`p_actual_winner`: simulated probability for the actual winner. `gap_mae_s`: error in "
        "gap to the winner for lead-lap finishers. `lap_pos_mae`: position error averaged over "
        "every lap of the race. Baselines have no distribution, so only position metrics apply.",
        "",
        "## Summary",
        "",
        md_table(summary),
        "",
        "## Simulator per race",
        "",
        md_table(per_race),
        "",
        f"## Held-out race: {held_name}",
        "",
        f"![simulated vs actual]({Path('..') / 'figures' / figure.name})" if figure else "",
        "",
        md_table(held, ".2f"),
        "",
        "## Calibrated simulator parameters (training seasons)",
        "",
        f"- Pit loss at {cfg['circuit']}: {pit:.1f} s (global {params.pit_loss_global:.1f} s, "
        f"stop-to-stop σ {params.pit_loss_sd:.1f} s); under SC/VSC ×{params.pit_loss_neutral_ratio:.2f} "
        "relative to the field.",
        f"- Safety-car lap {params.sc_lap_ratio:.2f}× race pace, VSC {params.vsc_lap_ratio:.2f}×.",
        f"- Overtaking (pairs < 1 s apart): intercept {params.overtake_a(cfg['circuit']):+.2f} at "
        f"{cfg['circuit']} (global {params.overtake_intercept_global:+.2f}), "
        f"{params.overtake_pace:+.2f} per s/lap pace advantage, {params.overtake_gap:+.2f} per s of gap; "
        f"held cars follow at {params.held_gap_low:.2f}–{params.held_gap_high:.2f} s.",
        f"- Lap-time variance split (from lap_net's out-of-season residuals): "
        f"{params.common_var_share:.0%} race-wide shift shared by the field, "
        f"{params.driver_var_share:.0%} race-long per-driver offset, "
        f"{1 - params.common_var_share - params.driver_var_share:.0%} lap-to-lap noise.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
