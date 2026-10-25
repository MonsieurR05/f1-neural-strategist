# API service

Thin FastAPI layer over `packages/strategist`: request validation, routing and caching. No
modelling logic lives here. Every response carries `provenance`: `historical` (what
happened) or `simulated` (counterfactual estimates).

| Endpoint | What it returns |
|---|---|
| `GET /health` | status, model tag, number of races |
| `GET /races?circuit=&year=` | race list with a `simulatable` flag (dry, representative qualifying) |
| `GET /races/{race_id}` | what happened: strategies, SC/VSC laps, positions and gaps to the leader per lap |
| `POST /simulate` | Monte Carlo replay from lap `start_lap`, optionally with edited strategies; finish distribution per driver and position/gap bands per lap |
| `POST /optimise` | ranked strategies for one driver from lap `start_lap`: expected position, win/podium/points odds, run-by-run comparison with the as-raced strategy, time budget and explanation |

`events`: `actual` replays the race's real safety cars and retirements (hindsight);
`sampled` draws them from history (a decision under uncertainty, the default for
`/optimise`).

## Run

From the repository root, after building the dataset and model artifacts
(`python -m strategist.sim.backtest` creates `artifacts/models/`):

```bash
pip install -e services/api[dev]
uvicorn app.main:app --app-dir services/api --reload
```

Interactive docs at http://localhost:8000/docs. Configuration: `configs/api.yaml`
(override the path with `F1S_API_CONFIG`). A full optimisation takes about 30 s on a CPU.

## Example

```bash
curl -X POST localhost:8000/optimise -H "content-type: application/json" \
     -d '{"race_id": "2024_01", "driver": "LEC", "start_lap": 1, "top": 5}'
```

## Deployment (Fly.io)

Live at https://f1-neural-strategist-api.fly.dev (health: `/health`, docs: `/docs`).

- Image: `services/api/Dockerfile` (CPU-only torch, pinned runtime deps, the strategist
  package without its pipeline/training extras, processed data, race results, track
  outlines and trained models; 390 MB compressed). The data and model files are local build
  artifacts, so deploy from a checkout where they have been built.
- Config: `fly.toml` at the repository root: one `shared-cpu-2x` machine with 1 GB in `lhr`
  that stops when idle (an optimisation peaks at ~650 MB), health check on `/health`, and
  `F1S_CORS_ORIGINS` set to the Vercel site only. `configs/api.deploy.yaml` is the public
  profile: lighter optimiser (~40 s on 2 vCPU when idle) and per-client quotas of 60
  simulations and 8 optimisations an hour, one optimisation at a time.

```bash
fly deploy --ha=false
```

## Tests

```bash
python -m pytest services/api/tests
```

They run against the real dataset and models and are skipped if those haven't been built.
