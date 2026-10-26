# Lap-time models: test season (2025)

_Generated 2026-10-26 10:57 UTC from commit `85c55ef` by `python -m strategist.eval.run`. Do not edit by hand._

**Task.** Forecast every clean lap of an unseen season from information available before the lap starts (qualifying pace, circuit, tyres, lap number, weather sample, gap to the car ahead). Trained on 2021, 2022, 2023, 2024 (68,465 laps); scored on 19,959 laps across 20 races.

**Final test.** Scored once. Models were refitted on every pre-test season with the settings fixed during development on the validation season; nothing was tuned on this season.

Excluded because qualifying was unrepresentative of race conditions (a race lap beat pole; usually wet qualifying): 2021_15 (Russian Grand Prix), 2022_09 (Canadian Grand Prix), 2022_10 (British Grand Prix), 2023_08 (Canadian Grand Prix), 2024_14 (Belgian Grand Prix), 2025_22 (Las Vegas Grand Prix).

**Metrics.** `mae_s`/`rmse_s`: absolute lap-time error. `within_mae_s`: error after removing each driver-race's mean error, i.e. what is left when comparing strategies for the same driver (the number that matters most for strategy). `stint_slope_mae`: error in the per-stint slope of lap time against tyre age (s per lap); `stint_slope_skill`: improvement over predicting no net slope (0 = none, 1 = perfect). `nll`/`coverage_90`: Gaussian uncertainty quality (σ from leave-one-season-out residuals; well calibrated coverage ≈ 0.90). `bias_s`: mean of predicted minus actual.

## Overall

| model | mae_s | rmse_s | within_mae_s | stint_slope_mae | stint_slope_skill | nll | coverage_90 | bias_s |
|---|---|---|---|---|---|---|---|---|
| circuit_pace | 1.175 | 1.500 | 0.832 | 0.052 | 0.000 | 1.901 | 0.896 | 0.143 |
| fuel_deg_linear | 1.085 | 1.384 | 0.512 | 0.052 | 0.013 | 1.970 | 0.870 | 0.142 |
| lightgbm | 0.986 | 1.273 | 0.533 | 0.051 | 0.036 | 1.865 | 0.879 | 0.209 |
| lap_net | 0.900 | 1.146 | 0.491 | 0.050 | 0.052 | 1.625 | 0.862 | 0.155 |

### Differences vs `lightgbm` (95% race-level paired bootstrap)

Negative = better than the reference. An interval spanning 0 means the difference is within race-to-race noise for this season.

| model | Δmae_s | Δwithin_mae_s | Δnll |
|---|---|---|---|
| circuit_pace | +0.188 [+0.132, +0.248] | +0.299 [+0.246, +0.359] | +0.035 [-0.320, +0.233] |
| fuel_deg_linear | +0.099 [+0.034, +0.172] | -0.022 [-0.050, +0.005] | +0.105 [+0.028, +0.201] |
| lap_net | -0.086 [-0.262, +0.075] | -0.042 [-0.068, -0.017] | -0.240 [-0.718, +0.081] |

## Seen vs unseen circuits

Circuits with no clean training laps: Montréal.

MAE (s):

| group | circuit_pace | fuel_deg_linear | lightgbm | lap_net | laps |
|---|---|---|---|---|---|
| seen circuit | 1.196 | 1.083 | 0.994 | 0.859 | 18763 |
| unseen circuit | 0.844 | 1.122 | 0.863 | 1.552 | 1196 |

Within-race MAE (s):

| group | circuit_pace | fuel_deg_linear | lightgbm | lap_net | laps |
|---|---|---|---|---|---|
| seen circuit | 0.843 | 0.518 | 0.539 | 0.494 | 18763 |
| unseen circuit | 0.662 | 0.416 | 0.446 | 0.439 | 1196 |

## Sakhir (MVP circuit)

| model | mae_s | within_mae_s | stint_slope_mae | stint_slope_skill | nll | coverage_90 | bias_s |
|---|---|---|---|---|---|---|---|
| circuit_pace | 0.859 | 0.714 | 0.073 | 0.000 | 1.589 | 0.983 | -0.286 |
| fuel_deg_linear | 0.700 | 0.513 | 0.064 | 0.120 | 1.356 | 0.959 | -0.285 |
| lightgbm | 0.634 | 0.471 | 0.087 | -0.194 | 1.319 | 0.966 | -0.299 |
| lap_net | 0.603 | 0.449 | 0.078 | -0.075 | 1.241 | 0.972 | 0.285 |

## By compound (within-race MAE, s)

| group | circuit_pace | fuel_deg_linear | lightgbm | lap_net | laps |
|---|---|---|---|---|---|
| HARD | 0.567 | 0.490 | 0.485 | 0.468 | 9007 |
| MEDIUM | 0.514 | 0.433 | 0.430 | 0.418 | 8242 |
| SOFT | 0.702 | 0.484 | 0.542 | 0.461 | 2710 |

## By tyre age (within-race MAE, s)

Long stints are where strategy counterfactuals extrapolate, so this is the band to watch.

| group | circuit_pace | fuel_deg_linear | lightgbm | lap_net | laps |
|---|---|---|---|---|---|
| 11–20 | 0.781 | 0.406 | 0.447 | 0.381 | 7213 |
| 1–10 | 0.905 | 0.496 | 0.536 | 0.479 | 6932 |
| 21–30 | 0.562 | 0.406 | 0.426 | 0.380 | 3968 |
| >30 | 0.496 | 0.470 | 0.474 | 0.460 | 1846 |

## Races with the largest absolute error

Large absolute but small within-race error points at a race-level pace offset (e.g. qualifying run in different conditions from the race) rather than a modelling failure.

Model: `lap_net`.

| race | laps | mae_s | within_mae_s |
|---|---|---|---|
| 2025_03 | 997 | 1.905 | 0.371 |
| 2025_10 | 1196 | 1.552 | 0.439 |
| 2025_16 | 914 | 1.471 | 0.478 |
| 2025_08 | 1262 | 1.238 | 1.162 |
| 2025_23 | 922 | 1.137 | 0.377 |
| 2025_15 | 1040 | 1.028 | 0.408 |

## `fuel_deg_linear` at Sakhir, in seconds

Coefficients scaled by the median qualifying lap at Sakhir (90.8 s). The fuel term multiplies laps remaining, so it is the time lost per lap of fuel still on board.

| fuel (s per lap of fuel) | deg SOFT (s / lap) | deg MEDIUM (s / lap) | deg HARD (s / lap) | MEDIUM vs SOFT fresh (s) | HARD vs SOFT fresh (s) |
|---|---|---|---|---|---|
| 0.068 | 0.119 | 0.055 | 0.072 | 0.028 | 0.220 |

## `lap_net` at Sakhir, in seconds

Fuel effect: 0.067 s per lap of fuel on board. Degradation relative to a 1-lap-old tyre (median conditions; compounds as nominated for Sakhir 2024: hard 1.5, medium 2, soft 3 on the hardness scale). Training data at Sakhir reaches 31 laps of tyre age; beyond that the curve is extrapolated by construction (non-decreasing, convex).

| compound | age 1 | age 5 | age 10 | age 15 | age 20 | age 25 | age 30 | age 35 | age 40 |
|---|---|---|---|---|---|---|---|---|---|
| SOFT | 0.00 | 0.25 | 0.61 | 1.03 | 1.52 | 2.08 | 2.70 | 3.38 | 4.08 |
| MEDIUM | 0.00 | 0.22 | 0.52 | 0.88 | 1.28 | 1.74 | 2.24 | 2.79 | 3.36 |
| HARD | 0.00 | 0.20 | 0.49 | 0.82 | 1.20 | 1.62 | 2.08 | 2.58 | 3.10 |

## `lightgbm` feature importance (share of total gain, %)

Boosting rounds: 49 (early stopping on the last training season).

| feature | gain_% |
|---|---|
| quali_gap_pct | 45.0 |
| fuel_frac | 28.3 |
| circuit | 7.5 |
| lap_number | 4.2 |
| track_temp | 3.7 |
| total_laps | 3.7 |
| tyre_life | 3.4 |
| laps_remaining | 1.2 |
| gap_ahead_s | 0.9 |
| grid_position | 0.8 |
