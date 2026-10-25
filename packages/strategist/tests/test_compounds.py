import numpy as np
import pandas as pd
import pytest
import torch

from strategist.data.compounds import add_hardness, hardness, load_table
from strategist.models.lapnet import LapNetModule
from tests.test_lapnet import random_batch


def test_hardness_scale_handles_2023_renumbering():
    assert hardness("C1", 2022) == 1.0 and hardness("C2", 2022) == 2.0
    assert hardness("C0", 2023) == 1.0          # the old C1, renamed
    assert hardness("C1", 2024) == 1.5          # new compound between old C1 and C2
    assert hardness("C6", 2025) == 6.0


def test_load_table_rejects_unordered_nominations(tmp_path):
    good = tmp_path / "good.csv"
    good.write_text("year,round,event_name,hard,medium,soft,source\n2024,1,Bahrain,C1,C2,C3,x\n")
    assert load_table(good).iloc[0][["h_HARD", "h_MEDIUM", "h_SOFT"]].tolist() == [1.5, 2.0, 3.0]
    bad = tmp_path / "bad.csv"
    bad.write_text("year,round,event_name,hard,medium,soft,source\n2024,1,Bahrain,C3,C2,C1,x\n")
    with pytest.raises(ValueError):
        load_table(bad)


def test_add_hardness_joins_race_and_lap_compound():
    table = pd.DataFrame({"year": [2024], "round": [1], "h_SOFT": [3.0], "h_MEDIUM": [2.0], "h_HARD": [1.5]})
    laps = pd.DataFrame({"year": [2024, 2024, 2024, 2023], "round": [1, 1, 1, 1],
                         "compound": ["SOFT", "HARD", "INTERMEDIATE", "SOFT"]})
    out = add_hardness(laps, table)
    assert out["hardness"].tolist()[:2] == [3.0, 1.5]
    assert np.isnan(out["hardness"].iloc[2]) and np.isnan(out["hardness"].iloc[3])


def test_softer_compound_is_never_slower_fresh_nor_wears_less_for_any_weights():
    torch.manual_seed(11)
    module = LapNetModule(n_circuits=5)
    for p in module.parameters():
        torch.nn.init.normal_(p, std=1.0)
    levels = torch.linspace(1.0, 6.0, 21)
    ages = torch.tensor([1.0, 10.0, 25.0, 40.0])
    for circuit in range(5):
        offs, degs = [], []
        for h in levels:
            b = random_batch(len(ages), ages)
            b.circuit = torch.full((len(ages),), circuit)
            b.hardness = torch.full((len(ages),), float(h))
            out = module(b)
            offs.append(out["offset"])
            degs.append(out["deg"])
        offs, degs = torch.stack(offs), torch.stack(degs)  # [levels, ages]
        assert torch.all(offs[1:] <= offs[:-1] + 1e-5)
        assert torch.all(degs[1:] >= degs[:-1] - 1e-5)
