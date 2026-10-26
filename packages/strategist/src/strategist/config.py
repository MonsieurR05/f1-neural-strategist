"""Config loading. Paths in configs are relative to the repository root (run commands from there)."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def training_seasons(split: dict[str, list[int]], evaluated: str) -> list[int]:
    """Seasons models are fitted on when scoring ``evaluated``. For the final test, models are
    refitted on every pre-test season (train + validation) with the settings chosen during
    development: what a real forecast of the test season would use. Nothing is tuned on it."""
    if evaluated == "test":
        return sorted(split["train"] + split["val"])
    return list(split["train"])
