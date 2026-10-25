"""Data and models loaded once at startup, plus race setups built from history."""

import copy
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from strategist.config import load_config
from strategist.models.features import prepare
from strategist.models.lapnet import LapTimeNet
from strategist.optimise.search import tyre_limits
from strategist.sim.history import setup_from_history
from strategist.sim.params import SimParams, lap_status
from strategist.sim.race import RaceSetup

MODEL_CAVEATS = [
    "Simulated outcomes are counterfactual estimates, not predictions of what would have happened.",
    "Tyres are modelled by their Pirelli compound (C0–C6 on an approximate hardness scale); "
    "differences between strategies of about a second are within the model's uncertainty.",
    "Other drivers keep their strategies; they do not react to the optimised driver.",
    "Dry races only. Tyre wear beyond the ages observed at the circuit is extrapolated.",
]


class AppState:
    def __init__(self, config_path: str | Path):
        self.cfg = load_config(config_path)
        data_cfg = load_config(self.cfg["data_config"])
        processed = Path(data_cfg["paths"]["processed"])
        self.track_dir = processed / "tracks"
        self.raw_root = Path(data_cfg["paths"]["raw"])
        self.laps = pd.read_parquet(processed / "laps.parquet")
        self.races = pd.read_parquet(processed / "races.parquet").set_index("race_id")
        model_dir, tag = Path(self.cfg["model_dir"]), self.cfg["model_tag"]
        torch.set_num_threads(int(os.environ.get("F1S_THREADS", self.cfg.get("threads", 4))))
        self.net = LapTimeNet.load(model_dir / f"lap_net_{tag}.pt")
        self.params = SimParams.load(model_dir / f"sim_params_{tag}.json")
        train_years = data_cfg["split"]["train"]
        self.train_laps = self.laps[self.laps["year"].isin(train_years)]
        self._setup = lru_cache(maxsize=64)(self._build_setup)

    def simulatable(self, race_id: str) -> bool:
        r = self.races.loc[race_id]
        return bool(not r["wet"] and r["quali_representative"])

    def race_laps(self, race_id: str) -> pd.DataFrame:
        return prepare(self.laps[self.laps["race_id"] == race_id])

    def results(self, race_id: str) -> pd.DataFrame:
        r = self.races.loc[race_id]
        return pd.read_parquet(self.raw_root / str(r["year"]) / f"{int(r['round']):02d}" / "results.parquet")

    def _build_setup(self, race_id: str, start_lap: int) -> tuple[RaceSetup, pd.DataFrame]:
        return setup_from_history(self.race_laps(race_id), self.results(race_id), start_lap)

    def setup(self, race_id: str, start_lap: int) -> tuple[RaceSetup, pd.DataFrame]:
        """A fresh copy (callers mutate events, retirements and strategies)."""
        setup, outcome = self._setup(race_id, start_lap)
        return copy.deepcopy(setup), outcome

    def tyre_limits(self, circuit: str) -> dict[str, int]:
        return tyre_limits(self.train_laps, circuit)

    def track(self, circuit: str) -> dict | None:
        path = self.track_dir / f"{circuit}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def events(self, race_id: str) -> pd.Series:
        return lap_status(self.laps[self.laps["race_id"] == race_id]).droplevel(0)


def config_path() -> str:
    return os.environ.get("F1S_API_CONFIG", "configs/api.yaml")


def finite_or_none(values) -> list[float | None]:
    return [None if v is None or not np.isfinite(v) else round(float(v), 3) for v in values]
