"""Evaluate lap-time models on a held-out season.

Usage (from the repository root):

    python -m strategist.eval.run --config configs/eval.yaml                  # validation season
    python -m strategist.eval.run --config configs/eval.yaml --split test --final

Task: forecast every clean lap of an unseen season from information available before the
lap starts. Models are fitted on the training seasons only. Each model's predictive
uncertainty is calibrated the same way: leave-one-season-out residuals within the training
seasons give a per-circuit Gaussian σ (in ratio units, scaled by the lap's ``q_s``).

The test season is for the final comparison only, so scoring it requires ``--final``.

Writes ``docs/reports/<report_name>_<split>.md`` and, under ``artifacts/eval/<split>/``,
per-lap predictions and a metrics JSON.
"""

import argparse
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from strategist.config import load_config
from strategist.eval.metrics import evaluate, within_mae
from strategist.models.baselines import FuelDegLinear, LightGBMLapTime, make_model
from strategist.models.features import clean_laps, prepare

log = logging.getLogger(__name__)

AGE_BANDS = [0, 10, 20, 30, np.inf]
AGE_LABELS = ["1–10", "11–20", "21–30", ">30"]


class CircuitSigma:
    """Per-circuit residual σ in ratio units, global fallback for sparse/unseen circuits."""

    def __init__(self, resid_ratio: pd.Series, circuits: pd.Series, min_laps: int):
        self.global_sigma = float(resid_ratio.std())
        counts = circuits.value_counts()
        per = resid_ratio.groupby(circuits).std()
        self.by_circuit = per[counts.reindex(per.index) >= min_laps]

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        sigma = df["circuit"].map(self.by_circuit).fillna(self.global_sigma)
        return sigma * df["q_s"]


def calibrate_sigma(name: str, params: dict, seed: int, train: pd.DataFrame,
                    min_laps: int) -> CircuitSigma:
    resid = []
    for season in sorted(train["year"].unique()):
        fold_train, fold_eval = train[train["year"] != season], train[train["year"] == season]
        mu = make_model(name, params, seed).fit(fold_train).predict(fold_eval)
        resid.append((fold_eval["lap_time_s"] - mu) / fold_eval["q_s"])
    resid = pd.concat(resid)
    return CircuitSigma(resid, train.loc[resid.index, "circuit"], min_laps)


def breakdown(df: pd.DataFrame, preds: dict[str, pd.DataFrame], by: pd.Series) -> pd.DataFrame:
    rows = []
    for model, p in preds.items():
        for group, idx in df.groupby(by).groups.items():
            g = df.loc[idx]
            rows.append({"group": group, "model": model, "laps": len(g),
                         "mae_s": float(np.mean(np.abs(g["lap_time_s"] - p.loc[idx, "mu"]))),
                         "within_mae_s": within_mae(g, g["lap_time_s"], p.loc[idx, "mu"])})
    return pd.DataFrame(rows)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    def fmt(v):
        if isinstance(v, float | np.floating):
            return "–" if np.isnan(v) else format(v, floatfmt)
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def pivot_metric(bd: pd.DataFrame, metric: str, order: list[str]) -> pd.DataFrame:
    t = bd.pivot(index="group", columns="model", values=metric)[order]
    laps = bd.drop_duplicates("group").set_index("group")["laps"]
    return t.assign(laps=laps).reset_index()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--final", action="store_true", help="required to score the test season")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.split == "test" and not args.final:
        parser.error("the test season is for the final comparison; pass --final to score it")

    cfg = load_config(args.config)
    data_cfg = load_config(cfg["data_config"])
    seed, models_cfg = cfg["seed"], cfg["models"]
    split = data_cfg["split"]
    processed = Path(data_cfg["paths"]["processed"])

    races = pd.read_parquet(processed / "races.parquet")
    excluded = races.loc[~races["quali_representative"] & ~races["wet"]
                         & races["year"].isin(split["train"] + split[args.split])]
    excluded = [f"{r.race_id} ({r.event_name})" for r in excluded.itertuples()]
    laps = prepare(clean_laps(pd.read_parquet(processed / "laps.parquet")))
    train = laps[laps["year"].isin(split["train"])]
    held = laps[laps["year"].isin(split[args.split])]
    seen = set(train["circuit"])
    log.info("train %d laps (%s), %s %d laps (%s)", len(train), split["train"], args.split,
             len(held), split[args.split])

    preds, metrics, fitted = {}, {}, {}
    for name in models_cfg:
        params = models_cfg[name]
        log.info("%s: calibrating σ (leave-one-season-out)", name)
        sigma = calibrate_sigma(name, params, seed, train, cfg["sigma"]["min_circuit_laps"])
        log.info("%s: fitting on all training seasons", name)
        model = make_model(name, params, seed).fit(train)
        mu, sd = model.predict(held), sigma(held)
        preds[name] = pd.DataFrame({"mu": mu, "sigma": sd})
        metrics[name] = evaluate(held, held["lap_time_s"], mu, sd)
        fitted[name] = model
        log.info("%s: MAE %.3f s, within-race MAE %.3f s", name, metrics[name]["mae_s"],
                 metrics[name]["within_mae_s"])

    out_dir = Path(cfg["artifacts"]) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_frame = held[["race_id", "circuit", "driver", "lap_number", "stint", "compound",
                       "tyre_life", "lap_time_s"]].copy()
    for name, p in preds.items():
        pred_frame[f"mu_{name}"], pred_frame[f"sigma_{name}"] = p["mu"], p["sigma"]
    pred_frame.to_parquet(out_dir / "predictions.parquet")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report = render_report(cfg, args.split, split, train, held, seen, preds, metrics, fitted,
                           excluded)
    report_path = Path(cfg["report_dir"]) / f"{cfg['report_name']}_{args.split}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    log.info("wrote %s", report_path)
    return 0


def render_report(cfg, split_name, split, train, held, seen, preds, metrics, fitted,
                  excluded) -> str:
    order = list(preds)
    overall = pd.DataFrame(metrics).T[["mae_s", "rmse_s", "within_mae_s", "stint_slope_mae",
                                       "stint_slope_skill", "nll", "coverage_90", "bias_s"]]
    overall = overall.reset_index(names="model")

    circuit_group = held["circuit"].map(lambda c: "seen circuit" if c in seen else "unseen circuit")
    focus = cfg["focus_circuit"]
    age_band = pd.cut(held["tyre_life"], AGE_BANDS, labels=AGE_LABELS).astype(str)
    by_race = breakdown(held, preds, held["race_id"])

    lines = [
        f"# Phase 1 baselines: {split_name} season ({', '.join(map(str, split[split_name]))})",
        "",
        f"_Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC from commit `{git_commit()}` by "
        "`python -m strategist.eval.run`. Do not edit by hand._",
        "",
        "**Task.** Forecast every clean lap of an unseen season from information available "
        "before the lap starts (qualifying pace, circuit, tyres, lap number, weather sample, gap "
        f"to the car ahead). Trained on {', '.join(map(str, split['train']))} "
        f"({len(train):,} laps); scored on {len(held):,} laps across "
        f"{held['race_id'].nunique()} races.",
        "",
        "Excluded because qualifying was unrepresentative of race conditions (a race lap beat "
        f"pole; usually wet qualifying): {', '.join(excluded) or 'none'}.",
        "",
        "**Metrics.** `mae_s`/`rmse_s`: absolute lap-time error. `within_mae_s`: error after "
        "removing each driver-race's mean error, i.e. what is left when comparing strategies for "
        "the same driver (the number that matters most for strategy). `stint_slope_mae`: error "
        "in the per-stint slope of lap time against tyre age (s per lap); `stint_slope_skill`: "
        "improvement over predicting no net slope (0 = none, 1 = perfect). `nll`/`coverage_90`: "
        "Gaussian uncertainty quality (σ from leave-one-season-out residuals; well calibrated "
        "coverage ≈ 0.90). `bias_s`: mean of predicted minus actual.",
        "",
        "## Overall",
        "",
        md_table(overall),
        "",
        "## Seen vs unseen circuits",
        "",
        "Circuits with no clean training laps: "
        f"{', '.join(sorted(set(held['circuit']) - seen)) or 'none'}.",
        "",
        "MAE (s):",
        "",
        md_table(pivot_metric(breakdown(held, preds, circuit_group), "mae_s", order)),
        "",
        "Within-race MAE (s):",
        "",
        md_table(pivot_metric(breakdown(held, preds, circuit_group), "within_mae_s", order)),
        "",
        f"## {focus} (MVP circuit)",
        "",
    ]
    focus_rows = held["circuit"] == focus
    if focus_rows.any():
        fh = held[focus_rows]
        focus_metrics = pd.DataFrame({m: evaluate(fh, fh["lap_time_s"], preds[m].loc[fh.index, "mu"],
                                                  preds[m].loc[fh.index, "sigma"])
                                      for m in order}).T
        lines += [md_table(focus_metrics[["mae_s", "within_mae_s", "stint_slope_mae",
                                          "stint_slope_skill", "nll", "coverage_90", "bias_s"]]
                           .rename_axis("model").reset_index()), ""]

    lines += [
        "## By compound (within-race MAE, s)",
        "",
        md_table(pivot_metric(breakdown(held, preds, held["compound"]), "within_mae_s", order)),
        "",
        "## By tyre age (within-race MAE, s)",
        "",
        "Long stints are where strategy counterfactuals extrapolate, so this is the band to watch.",
        "",
        md_table(pivot_metric(breakdown(held, preds, age_band), "within_mae_s", order)),
        "",
        "## Races with the largest absolute error",
        "",
        "Large absolute but small within-race error points at a race-level pace offset (e.g. "
        "qualifying run in different conditions from the race) rather than a modelling failure.",
        "",
    ]
    best = overall.sort_values("mae_s")["model"].iat[0]
    worst = (by_race[by_race["model"] == best].sort_values("mae_s", ascending=False)
             .head(6)[["group", "laps", "mae_s", "within_mae_s"]].rename(columns={"group": "race"}))
    lines += [f"Model: `{best}`.", "", md_table(worst), ""]

    if FuelDegLinear.name in fitted and focus in fitted[FuelDegLinear.name].coef:
        coef = fitted[FuelDegLinear.name].coefficients()
        q_ref = float(train.loc[train["circuit"] == focus, "q_s"].median())
        c = coef.loc[focus]
        interp = pd.DataFrame([{
            "fuel (s per lap of fuel)": c["laps_remaining"] * q_ref,
            "deg SOFT (s / lap)": c["age_SOFT"] * q_ref,
            "deg MEDIUM (s / lap)": c["age_MEDIUM"] * q_ref,
            "deg HARD (s / lap)": c["age_HARD"] * q_ref,
            "MEDIUM vs SOFT fresh (s)": c["is_MEDIUM"] * q_ref,
            "HARD vs SOFT fresh (s)": c["is_HARD"] * q_ref,
        }])
        lines += [
            f"## `{FuelDegLinear.name}` at {focus}, in seconds",
            "",
            f"Coefficients scaled by the median qualifying lap at {focus} ({q_ref:.1f} s). The fuel "
            "term multiplies laps remaining, so it is the time lost per lap of fuel still on board.",
            "",
            md_table(interp),
            "",
        ]

    if LightGBMLapTime.name in fitted:
        lgbm = fitted[LightGBMLapTime.name]
        gain = pd.Series(lgbm.booster.feature_importance("gain"), index=lgbm.FEATURES)
        gain = (100 * gain / gain.sum()).sort_values(ascending=False).head(10)
        lines += [
            f"## `{LightGBMLapTime.name}` feature importance (share of total gain, %)",
            "",
            f"Boosting rounds: {lgbm.rounds} (early stopping on the last training season).",
            "",
            md_table(gain.rename("gain_%").rename_axis("feature").reset_index(), ".1f"),
            "",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
