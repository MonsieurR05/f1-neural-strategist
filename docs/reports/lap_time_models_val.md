# Lap-time models: val season (2024)

_Generated 2026-10-26 10:56 UTC from commit `85c55ef` by `python -m strategist.eval.run`. Do not edit by hand._

**Task.** Forecast every clean lap of an unseen season from information available before the lap starts (qualifying pace, circuit, tyres, lap number, weather sample, gap to the car ahead). Trained on 2021, 2022, 2023 (48,496 laps); scored on 19,969 laps across 20 races.

Excluded because qualifying was unrepresentative of race conditions (a race lap beat pole; usually wet qualifying): 2021_15 (Russian Grand Prix), 2022_09 (Canadian Grand Prix), 2022_10 (British Grand Prix), 2023_08 (Canadian Grand Prix), 2024_14 (Belgian Grand Prix).

**Metrics.** `mae_s`/`rmse_s`: absolute lap-time error. `within_mae_s`: error after removing each driver-race's mean error, i.e. what is left when comparing strategies for the same driver (the number that matters most for strategy). `stint_slope_mae`: error in the per-stint slope of lap time against tyre age (s per lap); `stint_slope_skill`: improvement over predicting no net slope (0 = none, 1 = perfect). `nll`/`coverage_90`: Gaussian uncertainty quality (σ from leave-one-season-out residuals; well calibrated coverage ≈ 0.90). `bias_s`: mean of predicted minus actual.

## Overall

| model | mae_s | rmse_s | within_mae_s | stint_slope_mae | stint_slope_skill | nll | coverage_90 | bias_s |
|---|---|---|---|---|---|---|---|---|
| circuit_pace | 1.166 | 1.540 | 0.852 | 0.051 | 0.000 | 1.994 | 0.886 | -0.360 |
| fuel_deg_linear | 0.922 | 1.271 | 0.533 | 0.052 | -0.020 | 1.775 | 0.891 | -0.342 |
| lightgbm | 0.915 | 1.275 | 0.519 | 0.054 | -0.061 | 1.831 | 0.892 | -0.267 |
| lap_net | 0.916 | 1.274 | 0.538 | 0.055 | -0.072 | 1.660 | 0.909 | -0.254 |

### Differences vs `lightgbm` (95% race-level paired bootstrap)

Negative = better than the reference. An interval spanning 0 means the difference is within race-to-race noise for this season.

| model | Δmae_s | Δwithin_mae_s | Δnll |
|---|---|---|---|
| circuit_pace | +0.251 [+0.156, +0.341] | +0.333 [+0.273, +0.393] | +0.163 [+0.061, +0.250] |
| fuel_deg_linear | +0.007 [-0.069, +0.080] | +0.014 [-0.001, +0.032] | -0.056 [-0.221, +0.070] |
| lap_net | +0.001 [-0.043, +0.040] | +0.019 [-0.024, +0.086] | -0.171 [-0.653, +0.143] |

## Seen vs unseen circuits

Circuits with no clean training laps: Imola, Shanghai.

MAE (s):

| group | circuit_pace | fuel_deg_linear | lightgbm | lap_net | laps |
|---|---|---|---|---|---|
| seen circuit | 1.193 | 0.946 | 0.942 | 0.942 | 18043 |
| unseen circuit | 0.917 | 0.696 | 0.668 | 0.673 | 1926 |

Within-race MAE (s):

| group | circuit_pace | fuel_deg_linear | lightgbm | lap_net | laps |
|---|---|---|---|---|---|
| seen circuit | 0.870 | 0.536 | 0.522 | 0.533 | 18043 |
| unseen circuit | 0.681 | 0.507 | 0.494 | 0.585 | 1926 |

## Sakhir (MVP circuit)

| model | mae_s | within_mae_s | stint_slope_mae | stint_slope_skill | nll | coverage_90 | bias_s |
|---|---|---|---|---|---|---|---|
| circuit_pace | 0.955 | 0.915 | 0.052 | 0.000 | 1.687 | 0.990 | -0.037 |
| fuel_deg_linear | 0.454 | 0.366 | 0.045 | 0.123 | 1.330 | 0.994 | 0.077 |
| lightgbm | 0.422 | 0.365 | 0.058 | -0.111 | 1.354 | 0.993 | 0.097 |
| lap_net | 0.470 | 0.333 | 0.038 | 0.269 | 1.217 | 0.991 | 0.270 |

## By compound (within-race MAE, s)

| group | circuit_pace | fuel_deg_linear | lightgbm | lap_net | laps |
|---|---|---|---|---|---|
| HARD | 0.618 | 0.496 | 0.483 | 0.486 | 11949 |
| MEDIUM | 0.539 | 0.457 | 0.443 | 0.491 | 6698 |
| SOFT | 0.815 | 0.463 | 0.474 | 0.446 | 1322 |

## By tyre age (within-race MAE, s)

Long stints are where strategy counterfactuals extrapolate, so this is the band to watch.

| group | circuit_pace | fuel_deg_linear | lightgbm | lap_net | laps |
|---|---|---|---|---|---|
| 11–20 | 0.795 | 0.415 | 0.422 | 0.409 | 6901 |
| 1–10 | 0.840 | 0.483 | 0.462 | 0.477 | 7345 |
| 21–30 | 0.577 | 0.412 | 0.413 | 0.403 | 3647 |
| >30 | 0.574 | 0.524 | 0.536 | 0.523 | 2076 |

## Races with the largest absolute error

Large absolute but small within-race error points at a race-level pace offset (e.g. qualifying run in different conditions from the race) rather than a modelling failure.

Model: `lightgbm`.

| race | laps | mae_s | within_mae_s |
|---|---|---|---|
| 2024_08 | 1171 | 2.750 | 0.971 |
| 2024_10 | 1200 | 1.766 | 0.522 |
| 2024_23 | 714 | 1.511 | 0.694 |
| 2024_16 | 926 | 1.392 | 0.489 |
| 2024_13 | 1254 | 1.347 | 0.589 |
| 2024_15 | 1353 | 0.805 | 0.413 |

## `fuel_deg_linear` at Sakhir, in seconds

Coefficients scaled by the median qualifying lap at Sakhir (91.3 s). The fuel term multiplies laps remaining, so it is the time lost per lap of fuel still on board.

| fuel (s per lap of fuel) | deg SOFT (s / lap) | deg MEDIUM (s / lap) | deg HARD (s / lap) | MEDIUM vs SOFT fresh (s) | HARD vs SOFT fresh (s) |
|---|---|---|---|---|---|
| 0.068 | 0.122 | 0.052 | 0.063 | 0.062 | 0.310 |

## `lap_net` at Sakhir, in seconds

Fuel effect: 0.076 s per lap of fuel on board. Degradation relative to a 1-lap-old tyre (median conditions; compounds as nominated for Sakhir 2023: hard 1.5, medium 2, soft 3 on the hardness scale). Training data at Sakhir reaches 31 laps of tyre age; beyond that the curve is extrapolated by construction (non-decreasing, convex).

| compound | age 1 | age 5 | age 10 | age 15 | age 20 | age 25 | age 30 | age 35 | age 40 |
|---|---|---|---|---|---|---|---|---|---|
| SOFT | 0.00 | 0.42 | 0.96 | 1.56 | 2.19 | 2.86 | 3.56 | 4.29 | 5.03 |
| MEDIUM | 0.00 | 0.35 | 0.81 | 1.32 | 1.86 | 2.43 | 3.02 | 3.64 | 4.28 |
| HARD | 0.00 | 0.33 | 0.77 | 1.24 | 1.76 | 2.29 | 2.85 | 3.43 | 4.03 |

## `lightgbm` feature importance (share of total gain, %)

Boosting rounds: 140 (early stopping on the last training season).

| feature | gain_% |
|---|---|
| quali_gap_pct | 47.6 |
| fuel_frac | 15.3 |
| circuit | 11.1 |
| lap_number | 9.5 |
| track_temp | 4.3 |
| total_laps | 3.6 |
| tyre_life | 2.7 |
| grid_position | 2.1 |
| gap_ahead_s | 1.2 |
| laps_remaining | 0.9 |
