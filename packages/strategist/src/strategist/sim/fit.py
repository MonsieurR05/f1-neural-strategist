"""Fit the models the app serves: lap_net and simulator parameters on every season in the data.

Run after the final test season has been scored; the result has no held-out season left, so
its accuracy is the one reported for the 2025 test (models fitted on 2021-2024).

    python -m strategist.sim.fit --config configs/sim.yaml
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from strategist.config import load_config
from strategist.sim.backtest import fitted_models

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="configs/sim.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    data_cfg, eval_cfg = load_config(cfg["data_config"]), load_config(cfg["eval_config"])
    processed = Path(data_cfg["paths"]["processed"])
    laps_raw = pd.read_parquet(processed / "laps.parquet")
    races = pd.read_parquet(processed / "races.parquet")
    seasons = sorted({y for years in data_cfg["split"].values() for y in years})
    fitted_models(cfg, eval_cfg, laps_raw, races, seasons)
    log.info("models for %d-%d in %s", seasons[0], seasons[-1], cfg["model_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
