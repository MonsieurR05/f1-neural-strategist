"""Evaluate in-race adaptation: forecast the rest of a race from the laps already run.

Usage (from the repository root):

    python -m strategist.eval.inrace --config configs/inrace.yaml

For every race of the evaluation season and every cutoff (a share of race distance), each
model forecasts every clean lap after the cutoff from the clean laps up to it, given the
strategy each driver actually ran. All models start from the same lap_net forecast (fitted on
the training seasons) and predict its residual, so errors are the same whether stated as lap
times or residuals. Writes ``docs/reports/in_race_<split>.md``.
"""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from strategist.config import load_config, training_seasons
from strategist.eval.metrics import gaussian_nll, interval_coverage, mae, stint_slope_mae
from strategist.eval.run import git_commit, md_table, paired_bootstrap
from strategist.models.features import clean_laps, prepare
from strategist.models.inrace import (InRaceGP, InRaceGRU, green_flag_laps, known_at_cutoff,
                                       oos_frame, residual_frame)
from strategist.sim.backtest import fitted_models

log = logging.getLogger(__name__)

MODELS = ["lap_net", "driver_mean", "in_race_gp", "in_race_gru"]
LABELS = {
    "lap_net": "lap_net (pre-race, no update)",
    "driver_mean": "driver's mean error so far",
    "in_race_gp": "in-race GP",
    "in_race_gru": "in-race GRU",
}
HORIZONS = [(1, 5, "1–5 laps ahead"), (6, 15, "6–15"), (16, 40, "16–40"), (41, 999, ">40")]


def forecast(held: pd.DataFrame, flags: pd.DataFrame, cutoffs: list[float], gp: InRaceGP,
             gru: InRaceGRU) -> pd.DataFrame:
    """One row per (race, cutoff, future lap): residual and each model's mean/σ, in seconds.
    Future laps carry restart inputs as known at the cutoff (no further neutralisation)."""
    parts = []
    for race_id, g in held.groupby("race_id", sort=True):
        race_flags = flags[flags["race_id"] == race_id]
        total = int(g["total_laps"].iloc[0])
        for frac in cutoffs:
            k = int(round(frac * total))
            obs, target = g[g["lap_number"] <= k], g[g["lap_number"] > k]
            if target.empty:
                continue
            blind = known_at_cutoff(target.drop(columns="resid"), race_flags, k)
            q = target["q_s"].to_numpy() / 100.0
            out = target[["race_id", "driver", "stint", "lap_number", "tyre_life", "q_s"]].copy()
            out["cutoff"], out["k"] = frac, k
            out["y"] = target["resid"].to_numpy() * q
            out["mu_lap_net"], out["sigma_lap_net"] = 0.0, target["lapnet_sigma"].to_numpy() * q
            mean_so_far = obs.groupby("driver")["resid"].mean()
            out["mu_driver_mean"] = target["driver"].map(mean_so_far).fillna(0.0).to_numpy() * q
            out["sigma_driver_mean"] = np.nan
            for name, model in (("in_race_gp", gp), ("in_race_gru", gru)):
                mu, sigma = model.predict(obs, blind)
                out[f"mu_{name}"], out[f"sigma_{name}"] = mu * q, sigma * q
            current = obs.sort_values("lap_number").groupby("driver")["stint"].last()
            out["continuing"] = (target["driver"].map(current) == target["stint"]).to_numpy()
            out["ahead"] = target["lap_number"].to_numpy() - k
            parts.append(out)
    return pd.concat(parts, ignore_index=True)


def score(rows: pd.DataFrame) -> dict[str, float]:
    out = {}
    for m in MODELS:
        out[f"{m}|mae"] = mae(rows["y"], rows[f"mu_{m}"])
        sig = rows[f"sigma_{m}"]
        out[f"{m}|nll"] = gaussian_nll(rows["y"], rows[f"mu_{m}"], sig) if sig.notna().all() else np.nan
        out[f"{m}|cov"] = interval_coverage(rows["y"], rows[f"mu_{m}"], sig) if sig.notna().all() else np.nan
    return out


def metric_table(groups: dict[str, pd.DataFrame], metric: str) -> pd.DataFrame:
    rows = []
    for label, g in groups.items():
        s = score(g)
        rows.append({"": label, **{LABELS[m]: s[f"{m}|{metric}"] for m in MODELS}, "laps": f"{len(g):,}"})
    return pd.DataFrame(rows)


def rest_of_stint(rows: pd.DataFrame) -> pd.DataFrame:
    """Wear-rate error over the remaining laps of stints in progress at the cutoff."""
    cont = rows[rows["continuing"]].copy()
    cont["race_id"] = cont["race_id"] + "@" + cont["cutoff"].astype(str)  # one case per cutoff
    out = []
    for m in MODELS:
        err, n = stint_slope_mae(cont, cont["y"], cont[f"mu_{m}"])
        out.append({"model": LABELS[m], "slope_mae_s_per_lap": err, "stints": n})
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="configs/inrace.yaml")
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

    net, _ = fitted_models(cfg, eval_cfg, laps_raw, races, train_years)
    flags = green_flag_laps(laps_raw)
    with_flags = lambda df: df.merge(flags, on=["race_id", "lap_number"], how="left", validate="m:1")
    train = with_flags(prepare(clean_laps(laps_raw[laps_raw["year"].isin(train_years)])))
    held_laps = with_flags(prepare(clean_laps(laps_raw[laps_raw["year"].isin(held_years)])))
    if held_laps.empty:
        parser.error(f"no laps for {held_years}; ingest the season first")
    train_frame = oos_frame(net, train)
    mu_pct, sigma_pct = net.predict_ratio_pct(held_laps)
    held = residual_frame(held_laps, mu_pct).assign(lapnet_sigma=sigma_pct)
    log.info("train %d laps (%s, out-of-season residuals), %s %d laps (%s)", len(train_frame),
             train_years, args.split, len(held), held_years)

    gp = InRaceGP(**cfg["gp"]).fit(train_frame)
    gru = InRaceGRU(**cfg["gru"]).fit(train_frame)
    tag = f"{min(train_years)}-{max(train_years)}"
    Path(cfg["model_dir"], f"in_race_gp_{tag}.json").write_text(json.dumps(gp.state(), indent=1))

    rows = forecast(held, flags, cfg["cutoffs"], gp, gru)
    for m in MODELS:
        log.info("%s: MAE %.3f s", m, mae(rows["y"], rows[f"mu_{m}"]))

    report = render(cfg, args.split, train_years, held_years, rows, gp, gru)
    out = Path(cfg["report_dir"]) / f"{cfg['report_name']}_{args.split}.md"
    out.write_text(report, encoding="utf-8")
    log.info("wrote %s", out)
    return 0


def render(cfg, split_name, train_years, held_years, rows, gp: InRaceGP, gru: InRaceGRU) -> str:
    by_cutoff = {f"after {int(c * 100)}%": g for c, g in rows.groupby("cutoff")}
    by_cutoff["all cutoffs"] = rows
    held_bs = rows.rename(columns={"y": "lap_time_s"})
    preds = {m: pd.DataFrame({"mu": rows[f"mu_{m}"], "sigma": rows[f"sigma_{m}"]})
             for m in ("lap_net", "in_race_gp", "in_race_gru")}
    vs_lapnet = paired_bootstrap(held_bs, preds, "lap_net").drop(columns="Δwithin_mae_s")
    vs_gp = paired_bootstrap(held_bs, {m: preds[m] for m in ("in_race_gp", "in_race_gru")},
                             "in_race_gp").drop(columns="Δwithin_mae_s")
    stint_groups = {"stint in progress at the cutoff": rows[rows["continuing"]],
                    "after a pit stop still to come": rows[~rows["continuing"]]}
    horizon_groups = {label: rows[rows["ahead"].between(lo, hi)] for lo, hi, label in HORIZONS
                      if rows["ahead"].between(lo, hi).any()}
    p = gp.params
    lap = 90.0 / 100.0  # percent of the qualifying lap → seconds at a 90 s lap
    gp_rows = pd.DataFrame([
        {"term": "race (whole field)", "sd_s": p["race"] * lap, "note": ""},
        {"term": "after a start or restart", "sd_s": p["opening"] * lap, "note": f"mean {p['mean_opening'] * lap:+.2f} s on the first green lap, decay length {p['opening_len']:.1f} laps"},
        {"term": "race trend", "sd_s": p["trend"] * lap, "note": "per 50 laps"},
        {"term": "race drift", "sd_s": p["drift"] * lap, "note": f"correlation length {p['drift_len']:.0f} laps"},
        {"term": "compound in this race", "sd_s": p["compound"] * lap, "note": ""},
        {"term": "compound wear", "sd_s": p["compound_wear"] * lap, "note": "per 10 laps of tyre age"},
        {"term": "driver in this race", "sd_s": p["driver"] * lap, "note": ""},
        {"term": "driver pace swings", "sd_s": p["swing"] * lap, "note": f"correlation length {p['swing_len']:.1f} laps"},
        {"term": "stint", "sd_s": p["stint"] * lap, "note": ""},
        {"term": "stint wear", "sd_s": p["stint_wear"] * lap, "note": "per 10 laps of tyre age"},
        {"term": "lap noise", "sd_s": p["noise"] * lap, "note": ""},
        {"term": "mean wear error", "sd_s": p["mean_wear"] * lap, "note": "mean, per 10 laps (not an sd)"},
    ])
    lines = [
        f"# In-race adaptation: {split_name} season ({', '.join(map(str, held_years))})",
        "",
        f"_Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC from commit `{git_commit()}` by "
        "`python -m strategist.eval.inrace`. Do not edit by hand._",
        "",
        "**Task.** At a cutoff (a share of race distance), forecast every clean lap still to "
        "come from the clean laps run so far, given the strategy each driver actually ran. Every "
        f"model starts from lap_net's pre-race forecast (fitted on {', '.join(map(str, train_years))}) "
        "and predicts its error, so `lap_net` is the no-update baseline. Laps after the cutoff "
        "are forecast as if no further safety car comes (restarts after it are not known yet). "
        "In-race models are "
        "trained on lap_net's out-of-season residuals for the same seasons (each season predicted "
        f"by a lap_net fitted on the others). Scored on {rows[['race_id', 'k']].drop_duplicates().shape[0]} "
        f"race-cutoff cases across {rows['race_id'].nunique()} races, {len(rows):,} forecast laps "
        "(a lap counts once per cutoff it lies after).",
        "",
        "**Models.** `driver's mean error so far`: shift each driver's forecast by their mean "
        "lap_net error up to the cutoff (an analyst's correction; no uncertainty). `in-race GP`: "
        "Gaussian process with race, restart, trend, drift, compound, driver and stint terms "
        "(exact posterior). "
        f"`in-race GRU`: recurrent network over each driver's race so far ({gru.epochs} epochs, "
        "early stopping on the last training season).",
        "",
        "## MAE (s)",
        "",
        md_table(metric_table(by_cutoff, "mae")),
        "",
        "## Gaussian NLL",
        "",
        md_table(metric_table(by_cutoff, "nll")),
        "",
        "## 90% interval coverage",
        "",
        "Well calibrated ≈ 0.90.",
        "",
        md_table(metric_table(by_cutoff, "cov")),
        "",
        "### Differences vs `lap_net` (95% race-level paired bootstrap, all cutoffs)",
        "",
        md_table(vs_lapnet),
        "",
        "### GRU vs GP",
        "",
        md_table(vs_gp),
        "",
        "## By stint (MAE, s, all cutoffs)",
        "",
        "Laps of the stint a driver is on at the cutoff, and laps after a stop still to come "
        "(new tyres: only what the field has shown about that compound helps).",
        "",
        md_table(metric_table(stint_groups, "mae")),
        "",
        "## By horizon (MAE, s, all cutoffs)",
        "",
        md_table(metric_table(horizon_groups, "mae")),
        "",
        "## Rest-of-stint wear rate",
        "",
        "Error in the slope of lap time against tyre age over the remaining laps (≥ 8) of stints "
        "in progress at the cutoff, s/lap. `driver's mean error so far` shifts but doesn't tilt "
        "the forecast, so it matches lap_net here.",
        "",
        md_table(rest_of_stint(rows)),
        "",
        "## Fitted GP terms",
        "",
        "Standard deviations of each source of lap_net's out-of-season error, in seconds at a "
        "90 s lap (fitted in percent of the qualifying lap).",
        "",
        md_table(gp_rows, ".3f"),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
