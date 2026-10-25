"""Pirelli compound nominations per race and a hardness index comparable across seasons.

``configs/compounds.csv`` lists, per race, which Pirelli compounds were nominated as hard,
medium and soft (compiled from race articles: ``strategist.data.compound_sources``).

The C-numbers are not a fixed scale: in 2023 Pirelli renamed the old C1 to C0 and introduced
a new C1 between it and the C2, and 2025 added a softer C6. ``hardness`` puts them on one
ordinal scale (higher = softer):

    2021–22: C1 → 1, C2 → 2, … C5 → 5
    2023+:   C0 → 1 (the old C1), C1 → 1.5 (new, between), C2 → 2, … C6 → 6

The spacing is approximate (the lap-time model only relies on the order, through monotone
functions of hardness). Construction changes between seasons (18-inch tyres from 2022) are
not captured.
"""

from pathlib import Path

import numpy as np
import pandas as pd

LABELS = {"hard": "HARD", "medium": "MEDIUM", "soft": "SOFT"}
DEFAULT_HARDNESS = {"HARD": 2.0, "MEDIUM": 3.0, "SOFT": 4.0}  # typical nomination, if unknown


def hardness(compound: str, year: int) -> float:
    n = int(str(compound).strip().upper().lstrip("C"))
    if year >= 2023 and n <= 1:
        return 1.0 if n == 0 else 1.5
    return float(n)


def load_table(path: str | Path) -> pd.DataFrame:
    """One row per race: year, round, hardness of the nominated hard/medium/soft compounds."""
    table = pd.read_csv(path)
    for col, label in LABELS.items():
        table[f"h_{label}"] = [hardness(c, y) for c, y in zip(table[col], table["year"], strict=True)]
    bad = table[~((table["h_HARD"] < table["h_MEDIUM"]) & (table["h_MEDIUM"] < table["h_SOFT"]))]
    if not bad.empty:
        raise ValueError(f"compounds not ordered hard < medium < soft: {bad[['year', 'round']].values.tolist()}")
    return table[["year", "round", *(f"h_{l}" for l in LABELS.values())]]


def add_hardness(laps: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Join the race's nominated hardness (h_SOFT/h_MEDIUM/h_HARD) and the lap's own
    compound hardness (``hardness``; NaN for non-slick laps or races not in the table)."""
    laps = laps.merge(table, on=["year", "round"], how="left")
    own = np.full(len(laps), np.nan)
    for label in LABELS.values():
        own = np.where(laps["compound"].eq(label).to_numpy(), laps[f"h_{label}"].to_numpy(), own)
    laps["hardness"] = own
    return laps
