# Phase 1 baselines: val season (2024)

_Generated 2026-10-25 18:43 UTC from commit `19f65a3` by `python -m strategist.eval.run`. Do not edit by hand._

**Task.** Forecast every clean lap of an unseen season from information available before the lap starts (qualifying pace, circuit, tyres, lap number, weather sample, gap to the car ahead). Trained on 2021, 2022, 2023 (48,496 laps); scored on 19,969 laps across 20 races.

Excluded because qualifying was unrepresentative of race conditions (a race lap beat pole; usually wet qualifying): 2021_15 (Russian Grand Prix), 2022_09 (Canadian Grand Prix), 2022_10 (British Grand Prix), 2023_08 (Canadian Grand Prix), 2024_14 (Belgian Grand Prix).

**Metrics.** `mae_s`/`rmse_s`: absolute lap-time error. `within_mae_s`: error after removing each driver-race's mean error, i.e. what is left when comparing strategies for the same driver (the number that matters most for strategy). `stint_slope_mae`: error in the per-stint slope of lap time against tyre age (s per lap); `stint_slope_skill`: improvement over predicting no net slope (0 = none, 1 = perfect). `nll`/`coverage_90`: Gaussian uncertainty quality (σ from leave-one-season-out residuals; well calibrated coverage ≈ 0.90). `bias_s`: mean of predicted minus actual.

## Overall

| model | mae_s | rmse_s | within_mae_s | stint_slope_mae | stint_slope_skill | nll | coverage_90 | bias_s |
|---|---|---|---|---|---|---|---|---|
| circuit_pace | 1.166 | 1.540 | 0.852 | 0.051 | 0.000 | 1.994 | 0.886 | -0.360 |
| fuel_deg_linear | 0.922 | 1.271 | 0.533 | 0.052 | -0.020 | 1.775 | 0.891 | -0.342 |
| lightgbm | 0.927 | 1.288 | 0.521 | 0.055 | -0.069 | 1.850 | 0.889 | -0.264 |

## Seen vs unseen circuits

Circuits with no clean training laps: Imola, Shanghai.

MAE (s):

| group | circuit_pace | fuel_deg_linear | lightgbm | laps |
|---|---|---|---|---|
| seen circuit | 1.193 | 0.946 | 0.953 | 18043 |
| unseen circuit | 0.917 | 0.696 | 0.678 | 1926 |

Within-race MAE (s):

| group | circuit_pace | fuel_deg_linear | lightgbm | laps |
|---|---|---|---|---|
| seen circuit | 0.870 | 0.536 | 0.524 | 18043 |
| unseen circuit | 0.681 | 0.507 | 0.495 | 1926 |

## Sakhir (MVP circuit)

| model | mae_s | within_mae_s | stint_slope_mae | stint_slope_skill | nll | coverage_90 | bias_s |
|---|---|---|---|---|---|---|---|
| circuit_pace | 0.955 | 0.915 | 0.052 | 0.000 | 1.687 | 0.990 | -0.037 |
| fuel_deg_linear | 0.454 | 0.366 | 0.045 | 0.123 | 1.330 | 0.994 | 0.077 |
| lightgbm | 0.419 | 0.369 | 0.061 | -0.186 | 1.353 | 0.993 | 0.010 |

## By compound (within-race MAE, s)

| group | circuit_pace | fuel_deg_linear | lightgbm | laps |
|---|---|---|---|---|
| HARD | 0.618 | 0.496 | 0.483 | 11949 |
| MEDIUM | 0.539 | 0.457 | 0.446 | 6698 |
| SOFT | 0.815 | 0.463 | 0.477 | 1322 |

## By tyre age (within-race MAE, s)

Long stints are where strategy counterfactuals extrapolate, so this is the band to watch.

| group | circuit_pace | fuel_deg_linear | lightgbm | laps |
|---|---|---|---|---|
| 11–20 | 0.795 | 0.415 | 0.426 | 6901 |
| 1–10 | 0.840 | 0.483 | 0.462 | 7345 |
| 21–30 | 0.577 | 0.412 | 0.418 | 3647 |
| >30 | 0.574 | 0.524 | 0.534 | 2076 |

## Races with the largest absolute error

Large absolute but small within-race error points at a race-level pace offset (e.g. qualifying run in different conditions from the race) rather than a modelling failure.

Model: `fuel_deg_linear`.

| race | laps | mae_s | within_mae_s |
|---|---|---|---|
| 2024_08 | 1171 | 2.594 | 0.961 |
| 2024_10 | 1200 | 1.648 | 0.472 |
| 2024_13 | 1254 | 1.570 | 0.603 |
| 2024_15 | 1353 | 1.100 | 0.470 |
| 2024_16 | 926 | 1.029 | 0.496 |
| 2024_23 | 714 | 1.004 | 0.701 |

## `fuel_deg_linear` at Sakhir, in seconds

Coefficients scaled by the median qualifying lap at Sakhir (91.3 s). The fuel term multiplies laps remaining, so it is the time lost per lap of fuel still on board.

| fuel (s per lap of fuel) | deg SOFT (s / lap) | deg MEDIUM (s / lap) | deg HARD (s / lap) | MEDIUM vs SOFT fresh (s) | HARD vs SOFT fresh (s) |
|---|---|---|---|---|---|
| 0.068 | 0.122 | 0.052 | 0.063 | 0.062 | 0.310 |

## `lightgbm` feature importance (share of total gain, %)

Boosting rounds: 114 (early stopping on the last training season).

| feature | gain_% |
|---|---|
| quali_gap_pct | 47.4 |
| fuel_frac | 14.4 |
| lap_number | 10.8 |
| circuit | 10.3 |
| track_temp | 5.2 |
| total_laps | 2.9 |
| tyre_life | 2.9 |
| grid_position | 2.1 |
| laps_remaining | 1.3 |
| gap_ahead_s | 1.1 |
