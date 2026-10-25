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

- **Split:** by season, never random laps. Train 2021–2023 · validate 2024 ·
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

| Source | Seasons | Use |
|---|---|---|
| **FastF1** (community mirror) | 2021–2022 | laps, sectors, compound, tyre life, stints, pit in/out, positions, track status, weather, car X/Y |
| **OpenF1** | 2023–2025 | same live-timing feed, different shape; rebuilt into FastF1-style laps |
| **Jolpica** (Ergast successor) | all | classification, grid, qualifying Q1–Q3 times |

- **Scope:** dry races 2021–2025, all circuits, for lap-time training; Bahrain for
  simulation and UI in the MVP. 2026+ is a new regulation era — add as a separate
  era once it has enough data.
- **Volume (actual):** 114 races, 124.7k laps, **93.5k clean** — train 2021–23: 52.2k,
  val 2024: 20.6k, test 2025: 20.7k. 15 races are wet and excluded.
- **Cleaning:** nothing is deleted; each lap gets the first applicable `exclude_reason`:
  `wet_race` (≥10% inter/wet laps or ≥25% rain laps), `no_time`, `generated`, `lap1`,
  `pit_in`, `pit_out`, `tyre_data` (no compound, or a stint change not at a pit stop
  outside red flags), `neutralised` (SC/VSC/red), `non_slick`, `rain`, `damp` (within 2
  laps of anyone on inters), `inaccurate`, `slow_outlier` (>107% of race median).
- **Hand-built table:** event → Pirelli compound mapping (SOFT/MEDIUM/HARD → C1–C5).
- **Known confounders (documented, not hidden):** fuel load estimated; car pace
  shifts (2022 rules reset, upgrades) absorbed by team-season embeddings + quali
  prior; observed degradation includes deliberate tyre management.
- **Ingestion is offline only.** FastF1 is never called at request time.
- **Access (checked 2026-10-25):** livetiming.formula1.com now returns 401 without an
  F1 TV sign-in (FastF1 3.9 supports `F1_USERNAME`/`F1_PASSWORD`). Without it FastF1
  falls back to its community mirror, which covers **2021–2022 only** (44/44 races;
  0 for 2019–2020 and 2023–2025).
- **Sources used:** FastF1 mirror for 2021–2022, **OpenF1** for 2023–2025 (adapter rebuilds
  FastF1-style laps; track status replayed from race-control messages; line position
  derived from timing). Classification and qualifying from Jolpica. Both adapters write
  the same raw schema (`strategist.data.sources`), and `races.parquet` records the source.
- **OpenF1 data issues found and handled:**
  - stint lap ranges use different conventions between races (new stint on the in-lap
    in some, the out-lap in others), are sometimes split, and occasionally mislabel
    compounds (2025 Spa: inter laps labelled MEDIUM). Stint boundaries are therefore
    anchored on pit out-laps from timing; records only supply compound and starting age.
    This cut inconsistent driver-races from 177/1394 to 8.
  - the position feed lags the line crossing → position derived from timing order.
  - Jolpica lacks Q times for some events (2025 Miami) → best OpenF1 qualifying lap,
    skipping laps race control deleted; `quali_source` is recorded in meta.
- **Validation (all 114 races):** fresh-stint `TyreLife` starts at 1 in 99.4% (FastF1)
  and 100% (OpenF1) of stints; rebuilt SC/VSC laps match the real races; line order
  equals timing order in ≥99.9% of race-laps; Jolpica and lap-based qualifying times
  agree for 343/352 drivers (the rest are laps the stewards deleted). Official
  classification differs only by post-race penalties/DSQs, so **simulator backtests
  compare against road order at the line**, not the official classification.
- **API etiquette:** OpenF1 throttled to its free tier (30/min); FastF1's 500 calls/h
  budget is respected (`--wait-on-rate-limit`); OpenF1 and Jolpica responses are cached
  on disk, so rebuilding the dataset makes no network requests.

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

**MVP:** Bahrain 2021–2025 · dry with temperature variation · SC/VSC events ·
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
| 2026-10-25 | No F1 TV: seasons narrowed to 2021–2025 (FastF1 mirror 2021–22, OpenF1 2023–25); split train 2021–23 / val 2024 / test 2025 |
| 2026-10-25 | Phase 2 (`docs/MODEL_CARD.md`, `docs/reports/lap_time_models_val.md`): structured network as designed plus a bounded interaction term (no tyre-age input, so predictions stay non-decreasing in tyre age), 5-seed ensemble. Two-stage training: mean by Huber loss with epochs chosen on within-race MAE (a jointly trained σ head overfitted training-season noise and stopped the mean after 1 epoch), then σ head on leave-one-season-out residuals. **Identifiability rule** added after 2024 validation exposed a +4.7 s Lusail bias: race-level conditions (track temperature) may only enter terms shared across circuits. 2024: MAE 0.864 s (LightGBM 0.927; Δ −0.063, 95% race bootstrap [−0.133, −0.007]), within-race 0.527 (tie: Δ +0.005 [−0.020, +0.033]), NLL 1.607 (best), best on every metric at Bahrain. Bar not fully met: within-race tie, stint-slope skill still ≤ 0 overall, unseen circuits weaker (0.585). Validation was consulted for that one structural change only; 2025 test remains sealed. Next for the network: in-race adaptation (GRU) for per-stint degradation, circuit descriptors for unseen circuits |
| 2026-10-25 | Phase 1 (originally `docs/reports/phase1_baselines_val.md`, now `lap_time_models_val.md`): task = forecast every clean lap of an unseen season from pre-lap information, target `lap_time / driver's quali lap`. 2024 validation: analyst model MAE 0.922 s / within-race 0.533 s; LightGBM 0.927 / 0.521; circuit pace 1.166 / 0.852. No model predicts per-stint slopes better than "no net slope" (skill ≤ 0 overall). Races whose qualifying was unrepresentative (a race lap beat pole) are excluded from training and scoring. LightGBM settings chosen on an inner split (fit 2021–22, score 2023), never on 2024. Test season (2025) stays sealed until the final comparison. **Bar for the network:** within-race MAE < 0.52 s, positive stint-slope skill, lower error beyond 30 laps of tyre age, NLL < 1.78 with coverage ≈ 0.90 |
| 2026-10-25 | Phase 0 findings (`notebooks/01_bahrain_eda.ipynb`): fuel 0.06–0.08 s/lap; Bahrain deg ~linear 0.10–0.18 s/lap up to ~30 laps; compound offsets unidentifiable per race (CI 0.6–2.6 s) → pool across races; traffic +0.31 s within 0.5 s overall but ~+0.05 s at Bahrain → traffic term takes circuit. Before phase 2: build the C1–C5 compound table; flag stints beyond observed tyre ages as extrapolation in the UI |
