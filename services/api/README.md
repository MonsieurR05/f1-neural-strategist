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
(`python -m strategist.sim.fit` creates the served models in `artifacts/models/`):

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

## Deployment (Render)

Runs on Render's free instance (health: `/health`, docs: `/docs`).

- Image: `services/api/Dockerfile` (CPU-only torch, pinned runtime deps, the strategist
  package without its pipeline/training extras, processed data, race results, track
  outlines and trained models; 390 MB compressed). Render builds it from the repository, so
  the data and model files it copies are committed (`.gitignore` lets those paths through).
- Config: `render.yaml` at the repository root (a Render Blueprint): the free instance in
  Frankfurt, 512 MB and 0.1 CPU, which spins down after 15 idle minutes and takes about a
  minute to wake. Health check on `/health`, `F1S_CORS_ORIGINS` set to the Vercel site only.
  `configs/api.render.yaml` is the profile sized for it: 100 races per simulation, a coarse
  optimiser (~16 s on one full core, so a few minutes there), one optimisation at a time and
  per-client quotas of 30 simulations and 4 optimisations an hour. Measured locally the API
  holds ~450 MB and peaks at ~520 MB during an optimisation.
- Deploys: every push to `main` redeploys. First time: Render dashboard → New → Blueprint →
  this repository.

`fly.toml` and `configs/api.deploy.yaml` are the earlier Fly.io setup (one 2 vCPU / 1 GB
machine, full-size optimiser), kept for a paid host: `fly deploy --ha=false`.

## Tests

```bash
python -m pytest services/api/tests
```

They run against the real dataset and models and are skipped if those haven't been built.
