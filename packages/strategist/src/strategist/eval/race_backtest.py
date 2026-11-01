"""Backtest the race simulator on every dry race of a season, with and without in-race updates.

Usage (from the repository root):

    python -m strategist.eval.race_backtest --config configs/race_backtest.yaml

Every dry race at a circuit seen in training, with timing for every lap, is replayed from lap
1 and from shares of race distance, with the strategies, safety cars and retirements that actually happened. At each
cutoff the plain simulator (lap_net's pre-race pace) is compared with the same simulator
following samples of the in-race GP (``RaceSetup.pace_paths``) and with two orders that need
no simulation: the order frozen at the cutoff, and that order adjusted for the pit stops each
car still has to make. Writes ``docs/reports/race_backtest_<split>.md``.
"""

import argparse
import json
import logging
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from strategist.config import load_config, training_seasons
from strategist.eval.run import git_commit, md_table
from strategist.models.features import clean_laps, prepare
from strategist.models.inrace import (ROW_COLUMNS, InRaceGP, green_flag_laps, known_at_cutoff,
                                      residual_frame)
from strategist.sim.backtest import baseline_score, fitted_models, score
from strategist.sim.history import setup_from_history
from strategist.sim.race import RaceSetup, Simulator

log = logging.getLogger(__name__)

SIM, UPDATED, FROZEN, PIT_ADJUSTED = "simulator", "simulator + in-race GP", "frozen order", "pit-adjusted order"
MODELS = [UPDATED, SIM, PIT_ADJUSTED, FROZEN]
METRICS = ["pos_mae", "spearman", "coverage_80", "p_actual_winner", "gap_mae_s"]


def pit_adjusted_order(setup: RaceSetup, pit_loss: float) -> list[str]:
    """Order at the cutoff with each remaining stop's pit loss added: an analyst's estimate."""
    keys = {}
    for car, t in zip(setup.cars, setup.start_times, strict=True):
        stops = sum(lap > setup.start_lap for lap in car.strategy.pit_laps)
        keys[car.driver] = t + stops * pit_loss
    return sorted(keys, key=keys.get)


def pace_paths(gp: InRaceGP, obs: pd.DataFrame, race_laps: pd.DataFrame, flags: pd.DataFrame,
               setup: RaceSetup, runs: int, seed: int) -> np.ndarray:
    """Samples [runs, cars, laps + 1] of lap_net's error (percent) on the laps after the cutoff,
    from the clean laps up to it. Laps a car never ran stay at zero (it has retired by then)."""
    drivers = [c.driver for c in setup.cars]
    future = race_laps[(race_laps["lap_number"] > setup.start_lap) & race_laps["driver"].isin(drivers)]
    future = future.sort_values(["driver", "lap_number"]).copy()
    # tyre age is missing on a few laps; the neighbouring laps of the stint know it
    future["tyre_life"] = (future.groupby(["driver", "stint"])["tyre_life"]
                           .transform(lambda s: s.ffill().bfill()).fillna(1.0))
    columns = [c for c in ROW_COLUMNS if c not in ("since_green", "green_period")]
    target = known_at_cutoff(future[columns], flags, setup.start_lap)
    draws = gp.sample(obs, target, runs, np.random.default_rng(seed))
    paths = np.zeros((runs, len(drivers), setup.total_laps + 1))
    car = target["driver"].map({d: i for i, d in enumerate(drivers)}).to_numpy()
    paths[:, car, target["lap_number"].to_numpy().astype(int)] = draws
    return paths


def replay_race(race, laps_raw, held, flags, net, params, gp, cfg, raw_root) -> list[dict]:
    race_laps = prepare(laps_raw[laps_raw["race_id"] == race.race_id])
    results = pd.read_parquet(raw_root / str(race.year) / f"{race.round:02d}" / "results.parquet")
    race_held = held[held["race_id"] == race.race_id]
    race_flags = flags[flags["race_id"] == race.race_id]
    total = int(race_laps["lap_number"].max())
    rows = []

    def add(label, model, values):
        rows.append({"race": race.race_id, "event": race.event_name, "from": label, "model": model, **values})

    for label, k in [("lap 1", 1)] + [(f"{int(f * 100)}%", int(round(f * total))) for f in cfg["cutoffs"]]:
        setup, outcome = setup_from_history(race_laps, results, start_lap=k)
        sim = lambda s: Simulator(net, params, runs=cfg["runs"], seed=cfg["seed"], record_laps=False).run(s)
        add(label, SIM, score_finish(sim(setup), outcome))
        add(label, FROZEN, baseline_score([c.driver for c in setup.cars], outcome))
        add(label, PIT_ADJUSTED, baseline_score(pit_adjusted_order(setup, params.pit(setup.circuit)), outcome))
        if k > 1:
            obs = race_held[race_held["lap_number"] <= k]
            paths = pace_paths(gp, obs, race_laps, race_flags, setup, cfg["runs"], cfg["seed"])
            updated = replace(setup, pace_paths=paths, pace_noise_pct=gp.params["noise"])
            add(label, UPDATED, score_finish(sim(updated), outcome))
    log.info("%s %s done", race.race_id, race.event_name)
    return rows


def score_finish(result, outcome: pd.DataFrame) -> dict:
    """``sim.backtest.score`` for runs that keep only the finish (no per-lap position error)."""
    outcome = outcome.copy()
    outcome.attrs["positions_by_lap"] = pd.DataFrame()
    return {k: v for k, v in score(result, outcome).items() if k != "lap_pos_mae"}


def paired(scores: pd.DataFrame, a: str, b: str, metric: str = "pos_mae", n: int = 5000,
           seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a − b) over races and its 95% interval, resampling whole races (cutoffs of one
    race share its pace errors, so they are not independent)."""
    wide = scores[scores["model"].isin([a, b])].pivot_table(index="race", columns="model", values=metric)
    diff = (wide[a] - wide[b]).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(diff.mean()), float(lo), float(hi)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="configs/race_backtest.yaml")
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
    train_years = training_seasons(data_cfg["split"], args.split)
    held_years = data_cfg["split"][args.split]
    tag = f"{min(train_years)}-{max(train_years)}"

    net, params = fitted_models(cfg, eval_cfg, laps_raw, races, train_years)
    gp_path = Path(cfg["model_dir"]) / f"in_race_gp_{tag}.json"
    if not gp_path.exists():
        parser.error(f"{gp_path} is missing; run python -m strategist.eval.inrace first")
    gp = InRaceGP.from_state(json.loads(gp_path.read_text(encoding="utf-8")))

    flags = green_flag_laps(laps_raw)
    held_laps = prepare(clean_laps(laps_raw[laps_raw["year"].isin(held_years)])).merge(
        flags, on=["race_id", "lap_number"], how="left", validate="m:1")
    if held_laps.empty:
        parser.error(f"no laps for {held_years}; ingest the season first")
    held = residual_frame(held_laps, net.predict_ratio_pct(held_laps)[0])

    pool = races[races["year"].isin(held_years) & ~races["wet"] & races["quali_representative"]]
    laps_seen = laps_raw.groupby("race_id")["lap_number"].agg(["nunique", "max"])
    complete = pool["race_id"].map(laps_seen["nunique"] == laps_seen["max"]).fillna(False)
    seen = pool["circuit"].isin(net.circuit_index)
    left_out = [f"{r.event_name} (unseen circuit)" for r in pool[~seen].itertuples()]
    left_out += [f"{r.event_name} (laps missing at the source)" for r in pool[~complete].itertuples()]
    log.info("%d races, left out: %s", (seen & complete).sum(), left_out)
    rows = []
    for race in pool[seen & complete].sort_values("round").itertuples():
        rows += replay_race(race, laps_raw, held, flags, net, params, gp, cfg,
                            Path(data_cfg["paths"]["raw"]))
    scores = pd.DataFrame(rows)
    report = render(cfg, args.split, train_years, held_years, scores, left_out)
    out = Path(cfg["report_dir"]) / f"{cfg['report_name']}_{args.split}.md"
    out.write_text(report, encoding="utf-8")
    log.info("wrote %s", out)
    return 0


def render(cfg, split, train_years, held_years, scores, left_out) -> str:
    points = list(dict.fromkeys(scores["from"]))
    mid = scores[scores["from"] != "lap 1"]

    def pivot(metric):
        t = scores.pivot_table(index="from", columns="model", values=metric).reindex(points)
        return t.reindex(columns=[m for m in MODELS if m in t]).reset_index()

    all_cut = mid.groupby("model")[METRICS].mean().reindex(MODELS).reset_index()
    tests = []
    for label, frame in [("all cutoffs", mid)] + [(c, mid[mid["from"] == c]) for c in points[1:]]:
        for other in (SIM, PIT_ADJUSTED, FROZEN):
            d, lo, hi = paired(frame, UPDATED, other)
            tests.append({"from": label, "vs": other, "Δpos_mae": f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"})
    for other in (PIT_ADJUSTED, FROZEN):
        d, lo, hi = paired(mid, SIM, other)
        tests.append({"from": "plain simulator, all cutoffs", "vs": other, "Δpos_mae": f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"})
    d_sim, lo_sim, hi_sim = paired(mid, UPDATED, SIM)
    d_pit, lo_pit, hi_pit = paired(mid, UPDATED, PIT_ADJUSTED)
    verdict = "passes" if hi_sim < 0 and hi_pit < 0 else "does not pass"
    per_race = (mid[mid["model"].isin([UPDATED, SIM, PIT_ADJUSTED])]
                .pivot_table(index=["race", "event"], columns="model", values="pos_mae")
                .reindex(columns=[UPDATED, SIM, PIT_ADJUSTED]).reset_index())
    per_race["event"] = per_race["event"].str.replace(" Grand Prix", "")
    per_race["gain vs plain"] = per_race[UPDATED] - per_race[SIM]
    lap1 = scores[scores["from"] == "lap 1"]
    n_races = scores["race"].nunique()
    lines = [
        f"# Multi-race simulator backtest: {split} season ({', '.join(map(str, held_years))})",
        "",
        f"_Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC from commit `{git_commit()}` by "
        "`python -m strategist.eval.race_backtest`. Do not edit by hand._",
        "",
        f"**Setup.** {n_races} dry races at circuits seen in training"
        + (f" (left out: {', '.join(left_out)})" if left_out else "")
        + f", replayed from the actual order and gaps at each point with the strategies, safety cars "
        f"and retirements that actually happened, {cfg['runs']} Monte Carlo runs. lap_net, simulator "
        f"parameters and the GP are fitted on {min(train_years)}–{max(train_years)}. Simulated "
        "positions are estimates; `actual` is road order at the line, not the classification "
        "after penalties. A car that pits on the cutoff lap itself carries its in-lap loss only, "
        "for every model here.",
        "",
        "**Models.** `simulator`: lap_net's pre-race pace. `simulator + in-race GP`: follows "
        "samples of the GP's posterior given the clean laps up to the cutoff, assuming no further "
        "neutralisation in that forecast. `frozen order`: order at the cutoff. `pit-adjusted "
        "order`: order at the cutoff plus the circuit's pit loss per remaining stop.",
        "",
        "**Pass/fail test, fixed before this was run.** Pit-wall mode goes ahead only if the updated "
        "simulator beats both the plain simulator and the pit-adjusted order on position MAE over "
        "all cutoffs, with both race-level 95% intervals below zero. The GP was developed on this "
        "season, so a pass here is favourable to it; the 2026 test decides the final claim.",
        "",
        f"**Result: {verdict}.** Δ vs plain simulator {d_sim:+.3f} [{lo_sim:+.3f}, {hi_sim:+.3f}]; "
        f"vs pit-adjusted order {d_pit:+.3f} [{lo_pit:+.3f}, {hi_pit:+.3f}] places.",
        "",
        "## Position MAE by replay point",
        "",
        md_table(pivot("pos_mae")),
        "",
        "## Rank correlation with the actual finish",
        "",
        md_table(pivot("spearman")),
        "",
        "## All cutoffs",
        "",
        md_table(all_cut),
        "",
        "`coverage_80`: share of finishers inside the simulated 10–90% range (≈ 0.8 if calibrated). "
        "`p_actual_winner`: simulated probability for the actual winner. `gap_mae_s`: error in the "
        "gap to the winner for lead-lap finishers. Orders have no distribution.",
        "",
        "## Paired differences (95% race-level bootstrap)",
        "",
        "Negative = the first model is better.",
        "",
        md_table(pd.DataFrame(tests)),
        "",
        "## From lap 1, by race",
        "",
        "No in-race information yet, so this tests the plain simulator across circuits.",
        "",
        md_table(lap1.pivot_table(index="race", columns="model", values="pos_mae")
                 .reindex(columns=[SIM, PIT_ADJUSTED, FROZEN]).reset_index()),
        "",
        "## Mean of the cutoffs, by race (position MAE)",
        "",
        md_table(per_race),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
