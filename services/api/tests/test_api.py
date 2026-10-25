"""API tests against the real dataset and model artifacts (skipped if they haven't been built).

Run from the repository root:  python -m pytest services/api/tests
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs" / "api.yaml"
NEEDED = [ROOT / "data/processed/laps.parquet", ROOT / "artifacts/models/lap_net_2021-2023.pt",
          ROOT / "artifacts/models/sim_params_2021-2023.json"]

pytestmark = pytest.mark.skipif(not all(p.exists() for p in NEEDED),
                                reason="dataset/model artifacts not built")


@pytest.fixture(scope="module")
def client():
    import os
    os.chdir(ROOT)  # config paths are relative to the repository root
    from app.main import create_app
    with TestClient(create_app(str(CONFIG))) as c:
        yield c


def test_health_and_race_list(client):
    assert client.get("/health").json()["status"] == "ok"
    races = client.get("/races", params={"circuit": "Sakhir"}).json()
    ids = {r["race_id"]: r for r in races}
    assert "2024_01" in ids and ids["2024_01"]["simulatable"]


def test_race_detail_is_historical(client):
    detail = client.get("/races/2024_01").json()
    assert detail["provenance"] == "historical"
    ver = next(d for d in detail["drivers"] if d["driver"] == "VER")
    assert ver["finish_position"] == 1 and ver["strategy_label"].startswith("S-H-S")
    assert len(detail["positions_by_lap"]["VER"]) == 57


def test_track_outline_and_replay_timing(client):
    track = client.get("/circuits/Sakhir/track").json()
    assert len(track["points"]) == len(track["fraction"]) > 100 and track["fraction"][0] == 0.0
    assert client.get("/circuits/Nowhere/track").status_code == 404
    detail = client.get("/races/2024_01").json()
    assert 85 < detail["nominal_lap_s"] < 105 and len(detail["leader_lap_time_s"]) == 57


def test_unknown_and_wet_races(client):
    assert client.get("/races/1999_01").status_code == 404
    wet = next(r for r in client.get("/races").json() if r["wet"])
    assert client.post("/simulate", json={"race_id": wet["race_id"], "runs": 10}).status_code == 422


def test_simulate_with_strategy_override(client):
    base = {"race_id": "2024_01", "runs": 20}
    plain = client.post("/simulate", json=base).json()
    assert plain["provenance"] == "simulated" and len(plain["finish"]) == 20
    assert len(plain["laps"]["VER"]["median_pos"]) == 57  # laps 1..57
    override = {"VER": [{"compound": "SOFT", "end_lap": 17, "age_at_start": 3},
                        {"compound": "HARD", "end_lap": 57}]}
    one_stop = client.post("/simulate", json={**base, "strategies": override}).json()
    assert one_stop["strategies"]["VER"] == "S-H (stops 17)"


def test_override_cannot_rewrite_history(client):
    bad = {"VER": [{"compound": "HARD", "end_lap": 20}, {"compound": "SOFT", "end_lap": 57}]}
    r = client.post("/simulate", json={"race_id": "2024_01", "runs": 10, "strategies": bad})
    assert r.status_code == 422 and "history" in r.json()["detail"]
    one_compound = {"VER": [{"compound": "SOFT", "end_lap": 20, "age_at_start": 3},
                            {"compound": "SOFT", "end_lap": 57}]}
    r = client.post("/simulate", json={"race_id": "2024_01", "runs": 10, "strategies": one_compound})
    assert r.status_code == 422


def test_optimise_small(client):
    req = {"race_id": "2024_01", "driver": "VER", "start_lap": 30, "max_stops": 1,
           "screen_runs": 4, "refine_runs": 20, "top": 50}
    out = client.post("/optimise", json=req).json()
    assert out["provenance"] == "simulated" and out["candidates_evaluated"] > 0
    ranks = [s["expected_pos"] for s in out["strategies"]]
    assert ranks == sorted(ranks)
    assert sum(s["as_raced"] for s in out["strategies"]) == 1
    assert out["caveats"]
