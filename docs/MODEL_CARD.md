# Model card: structured probabilistic lap-time network (`lap_net`)

Code: `packages/strategist/src/strategist/models/lapnet.py` · Config: `configs/eval.yaml` ·
Evaluation: `docs/reports/lap_time_models_val.md` (generated; the source of truth for numbers).
Figures below are from the 2024 validation run of 2026-10-25.

## Purpose

Predict the time of a single racing lap, with uncertainty, from information known before the
lap starts. It is the building block of the race simulator: the simulator composes lap
predictions with pit-stop, safety-car and overtaking models to evaluate strategies that were
never actually raced. It is **not** a race-outcome predictor, and its outputs inside the
simulator are counterfactual estimates that the app labels as simulated.

## Inputs and output

| Group | Inputs |
|---|---|
| Pace anchor | driver's best qualifying lap `q_s` (target is lap time relative to it), gap to pole, missing-qualifying flag, grid position |
| Circuit | learned embedding; unseen circuits use a shared "unknown" embedding trained by 10% circuit dropout |
| Tyres | compound, tyre age, fresh/used set |
| Fuel | laps remaining (fuel load isn't published) |
| Traffic | gap to the car ahead at the start of the lap (capped at 10 s), race-leader flag |
| Conditions | track temperature |

Output: mean and standard deviation of the lap time in seconds (Gaussian).

## Architecture

Additive terms in percent of the qualifying lap, each a small MLP over the inputs it may use:

    base(circuit, quali pace, grid, race distance) + conditions(track temp, compound)
    + fuel_rate(circuit)·laps_remaining + offset(compound, circuit, fresh)
    + deg(tyre age | compound, circuit, fresh) + traffic(gap, leader, circuit)
    + interaction(all inputs except tyre age and temperature, bounded ±0.5 %)

Design guarantees, enforced by construction and unit-tested with random weights:

- **The predicted lap time never decreases with tyre age**, all other inputs fixed. Degradation
  is a non-negative mix of a linear term and softened hinges at 10/20/30 laps (it may steepen,
  a "cliff"), and the interaction term never sees tyre age. Counterfactual long stints
  therefore stay physically plausible.
- **Fuel effect ≥ 0** and linear in laps remaining, per circuit.
- **Identifiability rule:** race-level conditions (temperature) only enter terms shared by
  all circuits. With one to three races per circuit, a circuit-specific temperature effect
  would be confounded with everything else that differed between those races.

Uncertainty: a separate head trained on **leave-one-season-out residuals** (the mean model
is fitted on two training seasons and predicts the third), so σ reflects error on a new
season while still varying per lap (traffic, tyre age, compound). Five seeds are combined as
a Gaussian mixture.

## Training data and protocol

- 2021–2023 dry races, clean laps only (no pit, lap-1, safety-car, wet or damp laps; races whose
  qualifying was unrepresentative of race conditions excluded): 48,496 laps, 52 races,
  23 circuits.
- Mean: Huber loss, AdamW. Epoch count chosen by within-race MAE on 2023 when fitting on
  2021–22, then all models refitted on 2021–23.
- Validation: 2024 (19,969 laps, 20 races). Test: 2025, **not yet used**; it is reserved for
  the final model comparison.

## Results (2024 validation)

| Metric | lap_net | LightGBM | Linear deg (analyst) |
|---|---|---|---|
| Lap-time MAE | **0.864 s** | 0.927 s | 0.922 s |
| Within-race MAE (strategy-relevant) | 0.527 s | **0.521 s** | 0.533 s |
| Gaussian NLL | **1.607** | 1.850 | 1.775 |
| 90% interval coverage | 0.919 | 0.889 | 0.891 |
| Bahrain lap-time MAE / within-race MAE | **0.385 / 0.320 s** | 0.419 / 0.369 s | 0.454 / 0.366 s |
| Bahrain stint-slope skill | **+0.32** | −0.19 | +0.12 |

Against LightGBM, race-level paired bootstrap (95%): MAE −0.063 s [−0.133, −0.007] (better),
within-race MAE +0.005 s [−0.020, +0.033] (tie), NLL −0.243 [−0.721, +0.066] (likely better).

Learned Bahrain behaviour (median conditions): fuel 0.074 s per lap of fuel; degradation at 20
laps soft 2.6 s, medium 1.5 s, hard 1.7 s, steepening with age.

## Limitations

- **Per-stint degradation is mostly unpredictable before the race.** Across all circuits the
  stint-slope skill is ≤ 0 for every model, this one included: drivers manage tyres to plan,
  so a stint's degradation is driver- and stint-specific. Planned fix: in-race adaptation
  (a sequence model conditioned on the laps already run).
- **Unseen circuits are weaker** (within-race MAE 0.585 s vs 0.520 s on seen circuits). The
  "unknown" embedding is a weak prior; circuit descriptors (length, corners, speed) would
  generalise better.
- **Beyond observed tyre ages** (about 30 laps at Bahrain) the degradation curve is an
  extrapolation: monotone and convex by construction, not validated by data. The simulator
  should flag such stints.
- **Pace anchor:** requires a representative qualifying session. Wet or otherwise
  unrepresentative qualifying makes absolute predictions unreliable for that race.
- **Dry conditions only.** Wet races and laps are excluded from training.
- **Associational:** observed laps reflect teams' strategy choices and tyre management, and
  fuel load is a proxy. Effects are estimates, not measured physics.
- **Compound labels** (soft/medium/hard) map to different Pirelli compounds between events;
  a C1–C5 mapping is not yet used.
- One validation-driven change was made (the identifiability rule above); everything else was
  chosen on the inner split. The sealed 2025 test season will give the unbiased estimate.
