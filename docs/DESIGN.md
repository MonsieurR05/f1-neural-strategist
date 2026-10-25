# F1 Neural Race Strategist — Design

Status: **agreed, pre-implementation** · Last updated: 2026-10-25

## 1. Problem and objectives

**Problem.** Given the state of a race, which pit strategy gives a chosen driver the
best distribution of finishing positions, and why?

It is a decision under uncertainty driven by three things:

1. **Tyre degradation and fuel burn** — non-linear, circuit-, compound- and
   temperature-dependent.
2. **Interactions** — traffic, dirty air, undercut/overcut, overtaking difficulty.
3. **Stochastic events** — safety cars, VSCs, slow stops.

**Objectives (in priority order)**

1. A **learned lap-time model** that beats non-neural baselines on held-out races.
2. A **race simulator** that reproduces real results when replaying historical races
   with the strategies actually run (the credibility metric for every counterfactual).
3. A **strategy optimiser** returning ranked strategies with outcome distributions
   and component-level explanations.
4. An **interactive UI**: build, compare and watch strategies; "pit wall" mode to
   pause mid-race and ask "box now?".

**Non-goals:** a statistics dashboard, an LLM chatbot, a racing game.

## 2. Neural networks

### Why the network does not predict race outcomes directly

History contains one strategy per driver per race. A model trained on
"strategy → finishing position" learns correlations (front-runners 1-stop), not
causes. Instead the network learns the **building blocks** (lap time under given
conditions) and the simulator composes them into counterfactuals.

### Network A — probabilistic lap-time model (MVP, core)

- **Target:** lap time, predicted as mean and variance (Gaussian NLL). The variance
  drives Monte Carlo sampling.
- **Inputs**

  | Group | Features |
  |---|---|
  | Tyre | compound, tyre age, fresh/used set, stint number |
  | Fuel | laps remaining (fuel load is not published) |
  | Pace prior | qualifying gap to pole (known pre-race, no leakage) |
  | Conditions | track temperature, air temperature |
  | Context | circuit embedding, gap to car ahead, track status |
  | Identity | driver and team-season embeddings |

- **Architecture: structured additive ("grey-box") network**

  ```
  lap_time = base(circuit, pace_prior)
           + fuel_net(laps_remaining)
           + tyre_net(compound, age, temp, circuit)   # monotone non-decreasing in age
           + traffic_net(gap_ahead)
           + noise σ(·)
  ```

  Rationale: counterfactual strategies produce stints longer than observed; the
  monotonicity constraint prevents physically impossible extrapolation, and the
  additive terms give explanations for free.

- **Baselines** (all evaluated on the same held-out races):
  1. driver/circuit mean
  2. fuel-corrected linear degradation per compound (standard analyst model)
  3. LightGBM — may win on raw MAE; report honestly. The network should win on
     extrapolation, uncertainty calibration and low-data circuits.

- **Metrics:** MAE, RMSE, NLL, calibration (PIT histogram / interval coverage),
  stint degradation-slope error, error on long-stint extrapolation slices.

- **Split:** by season, never random laps. Train 2019–2023 · validate 2024 ·
  test 2025.

- **v2 — GRU in-race adaptation:** condition on the laps already run in the current
  stint to infer today's degradation. Test: predict the rest of a stint from its
  first *k* laps, with vs. without the sequence encoder.

### Network B — strategy value surrogate (phase 2)

`V(race_state, action) → expected finishing position`, trained on simulator
rollouts. Enables instant ranking in pit-wall mode. Evaluated against full Monte
Carlo: Spearman ρ, top-1 agreement, latency speed-up.

### Deliberately not neural

| Component | Method | Why |
|---|---|---|
| SC / VSC occurrence | empirical per-circuit hazard | too few events |
| Overtake probability | logistic regression on pace delta + circuit | small data, needs interpretability |
| Pit loss | per-circuit empirical distribution | it is a measurement |
| Strategy search | enumeration + Monte Carlo | action space is small; RL adds instability, no gain |

## 3. Data

| Source | Use | Notes |
|---|---|---|
| **FastF1** (primary) | laps, sectors, compound, tyre life, stints, pit in/out, positions, track status, weather, car X/Y | timing data from 2018; results via Jolpica (Ergast successor) |
| **OpenF1** | intervals, pit durations, race control | 2023+ only; later live mode |

- **Scope:** dry races 2019–2025, all circuits, for lap-time training; Bahrain for
  simulation and UI in the MVP. 2026+ is a new regulation era — add as a separate
  era once it has enough data.
- **Volume:** ~24k laps/season → ~90–110k usable laps after cleaning.
- **Cleaning:** drop wet races, lap 1, in/out laps, SC/VSC laps (modelled
  separately), laps flagged inaccurate.
- **Hand-built table:** event → Pirelli compound mapping (SOFT/MEDIUM/HARD → C1–C5).
- **Known confounders (documented, not hidden):** fuel load estimated; car pace
  shifts (2022 rules reset, upgrades) absorbed by team-season embeddings + quali
  prior; observed degradation includes deliberate tyre management.
- **Ingestion is offline only.** FastF1 is never called at request time.

**Why Bahrain:** always dry (night race), high degradation, varied strategies
(1–3 stops), overtaking is possible, on the calendar every season.

## 4. Simulator ↔ network

```
offline:  FastF1 → ingest/clean → Parquet → train lap-time model → ONNX artifact
                                         └→ fit pit loss, SC hazard, overtake model

online:   scenario → Simulator (lap-by-lap, vectorised over N Monte Carlo runs)
            per lap, per car:
              lap time ~ LapTimeModel(state)          ← Network A
              resolve traffic / overtakes             ← logistic model
              pit stops (pit loss + stationary-time distribution)
              sample SC/VSC (compresses gaps, cheaper stops)
          → finishing-position distributions, gaps, per-lap state

optimiser: enumerate legal strategies (two-compound rule) → simulate →
           rank by E[position], P(podium), risk
           phase 2: Network B pre-ranks, Monte Carlo confirms top-k
```

- **Backtest:** replay held-out Bahrain races with actual strategies; report
  position MAE, Spearman ρ of finishing order, race-time error.
- **Opponents:** MVP keeps historical strategies; phase 2 adds rule-based reactions
  (cover undercut, pit under SC); game-theoretic opponents are advanced.
- **External sanity check:** compare recommendations with Pirelli's published
  pre-race strategy predictions.
- **Provenance:** every API value carries `source: historical | simulated | estimated`;
  the UI renders simulated data distinctly (e.g. dashed). Car motion within a lap is
  interpolated and labelled as such.

## 5. User interface

1. **Scenario picker** — circuit, season, driver.
2. **Strategy builder** — stint timeline with draggable pit markers and compound
   chips; outcome updates live.
3. **Comparison** — strategies A/B/C: finishing-position histograms, gap-to-rival,
   degradation curves, time breakdown (deg / pit loss / traffic / SC).
4. **Race replay** — 2D SVG track map (outline from FastF1 X/Y) with animated cars,
   synced position and gap charts; historical vs simulated side by side.
5. **Pit wall mode** — scrub to any lap, inject an SC, ask "box or stay out?";
   ranked options, win probabilities, explanation from model components.

Rendering: SVG/Canvas + D3/visx. Three.js only as optional, contained polish.

## 6. Scope

**MVP:** Bahrain 2019–2025 · dry with temperature variation · SC/VSC events ·
Network A + 3 baselines with written evaluation · simulator + backtest report ·
Monte Carlo optimiser · builder, comparison, replay · deployed.

**Phase 2:** Network B + pit wall mode · GRU in-race adaptation · reactive opponents ·
3–5 more circuits.

**Advanced (optional):** rain/intermediates (experimental) · game-theoretic
opponents · OpenF1 live mode · RL policy benchmarked against the optimiser.

## 7. Architecture and roadmap

```
apps/web/              Next.js + TypeScript + Tailwind, D3/visx, SVG track
services/api/          FastAPI — thin: validation, routes, caching
packages/strategist/   installable Python package — the core
  data/                FastF1 ingest, cleaning, features → Parquet
  models/              lap-time net, baselines, value net, training
  sim/                 vectorised NumPy simulator, events, rules
  optimise/            strategy enumeration, Monte Carlo ranking
  eval/                metrics, backtests, reports
configs/               YAML experiment configs (seeded)
notebooks/             exploration only, never imported
docs/                  design, model card, evaluation reports
```

- Storage: Parquet + DuckDB. SQLite/Postgres only when users can save scenarios.
- Train in PyTorch, serve with ONNX Runtime (small CPU-only image).
- Deploy: Vercel (web) + Fly.io / Render / Railway (API).
- Reproducibility: configs + seeds + versioned artifacts; MLflow only if needed.
- Legal: unofficial, non-commercial; no F1 logos; disclaimer in UI and README.

| Phase | Est. weeks | Exit criterion |
|---|---|---|
| 0. Setup + data exploration | 1–2 | clean Parquet dataset; EDA notebook with degradation curves |
| 1. Eval harness + baselines | 1–2 | held-out metrics for all three baselines |
| 2. Lap-time network | 2–3 | beats baselines on a defined axis; model card |
| 3. Simulator + backtest | 2–3 | backtest report on held-out Bahrain races |
| 4. Optimiser + API | 1–2 | `/races`, `/simulate`, `/optimise` tested |
| 5. Frontend MVP | 3–4 | builder, comparison, replay; deployed |
| 6. Surrogate + pit wall | 3–4 | speed vs accuracy numbers against Monte Carlo |

## Decision log

| Date | Decision |
|---|---|
| 2026-10-25 | MVP circuit: Bahrain |
| 2026-10-25 | Seasons 2019–2025, all dry races for lap-time training; team-season embeddings absorb the 2022 rules change |
| 2026-10-25 | Rendering: 2D SVG + D3/visx; Three.js optional later |
| 2026-10-25 | No database in MVP (Parquet + DuckDB) |
| 2026-10-25 | Monorepo on GitHub, private until MVP is presentable |
