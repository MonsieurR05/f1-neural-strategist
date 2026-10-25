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
| Tyres | compound as Pirelli hardness (C0–C6 on one scale, from `configs/compounds.csv`), tyre age, fresh/used set |
| Fuel | laps remaining (fuel load isn't published) |
| Traffic | gap to the car ahead at the start of the lap (capped at 10 s), race-leader flag |
| Conditions | track temperature |

Output: mean and standard deviation of the lap time in seconds (Gaussian).

## Architecture

Additive terms in percent of the qualifying lap, each a small MLP over the inputs it may use:

    base(circuit, quali pace, grid, race distance) + conditions(track temp)
    + fuel_rate(circuit)·laps_remaining + offset(hardness | circuit, fresh)
    + deg(tyre age | hardness, circuit, fresh) + traffic(gap, leader, circuit)
    + interaction(all inputs except tyre age, temperature and compound, bounded ±0.5 %)

Design guarantees, enforced by construction and unit-tested with random weights:

- **The predicted lap time never decreases with tyre age**, all other inputs fixed. Degradation
  is a non-negative mix of a linear term and softened hinges at 10/20/30 laps (it may steepen,
  a "cliff"), and the interaction term never sees tyre age. Counterfactual long stints
  therefore stay physically plausible.
- **Fuel effect ≥ 0** and linear in laps remaining, per circuit.
- **Compound hardness:** compounds enter as the nominated Pirelli compound on one hardness
  scale (2021–22 C1–C5 → 1–5; 2023+ C0 → 1, the new C1 → 1.5, C2–C6 → 2–6). Fresh-tyre pace
  is non-increasing and every degradation weight non-decreasing in hardness, at every circuit
  and tyre state (non-negative combinations of increasing hardness bases; temperature and
  interaction terms don't see the compound). So a softer compound is never slower when fresh
  nor wears less, and within a race soft/medium/hard are ordered.
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
| Lap-time MAE | 0.916 s | **0.915 s** | 0.922 s |
| Within-race MAE (strategy-relevant) | 0.538 s | **0.519 s** | 0.533 s |
| Gaussian NLL | **1.660** | 1.831 | 1.775 |
| 90% interval coverage | 0.909 | 0.892 | 0.891 |
| Bahrain lap-time MAE / within-race MAE | 0.470 / **0.333 s** | **0.422** / 0.365 s | 0.454 / 0.366 s |
| Bahrain stint-slope skill | **+0.27** | −0.11 | +0.12 |

(LightGBM also receives the compound hardness.) Against LightGBM, race-level paired bootstrap
(95%): MAE +0.001 s [−0.043, +0.040], within-race MAE +0.019 s [−0.024, +0.086], NLL −0.171
[−0.653, +0.138]: within race-to-race noise for a single season.

History of the compound handling, for context: with raw labels the network scored MAE
0.864 s but believed the Bahrain medium was both faster and longer-lasting than the hard; a
label-ordering constraint fixed that (0.888 s) but still blended different Pirelli compounds
under one label; hardness from the nomination table fixed the blend (0.916 s on this metric,
with the decision-level improvements below). Single-season MAE differences of a few
hundredths are within noise; the decision-level checks are the ones that changed.

Learned 2024 Bahrain behaviour (median conditions; hard C1, medium C2, soft C3): degradation
after 20 laps hard 1.8 s, medium 1.9 s, soft 2.3 s; fresh-tyre pace soft fastest. Backtest
(2024 Bahrain, held out): position MAE 2.10, 80% coverage 0.90, P(actual winner) 0.52.
The optimiser rates Verstappen's winning soft–hard–soft within about a second of its best
plan.

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
- **Compound scale is approximate:** the hardness spacing is ordinal (the model only relies on
  order), construction changes between seasons (13-inch to 18-inch wheels in 2022) aren't
  modelled, and whether 2025's C1 equals 2024's C1 is assumed, not verified (affects six 2025
  races, all in the test season). Nominations come from Wikipedia race articles (112 of 114
  parsed; 2 resolved from F1/Pirelli previews; 18 cross-checked against F1/Pirelli, all agree).
- One validation-driven change was made (the identifiability rule above); everything else was
  chosen on the inner split. The sealed 2025 test season will give the unbiased estimate.
