# Multi-race simulator backtest: val season (2025)

_Generated 2026-11-01 19:03 UTC from commit `f70b138` by `python -m strategist.eval.race_backtest`. Do not edit by hand._

**Setup.** 18 dry races at circuits seen in training (left out: Canadian Grand Prix (unseen circuit), Miami Grand Prix (laps missing at the source)), replayed from the actual order and gaps at each point with the strategies, safety cars and retirements that actually happened, 500 Monte Carlo runs. lap_net, simulator parameters and the GP are fitted on 2021–2024. Simulated positions are estimates; `actual` is road order at the line, not the classification after penalties. A car that pits on the cutoff lap itself carries its in-lap loss only, for every model here.

**Models.** `simulator`: lap_net's pre-race pace. `simulator + in-race GP`: follows samples of the GP's posterior given the clean laps up to the cutoff, assuming no further neutralisation in that forecast. `frozen order`: order at the cutoff. `pit-adjusted order`: order at the cutoff plus the circuit's pit loss per remaining stop.

**Pass/fail test, fixed before this was run.** Pit-wall mode goes ahead only if the updated simulator beats both the plain simulator and the pit-adjusted order on position MAE over all cutoffs, with both race-level 95% intervals below zero. The GP was developed on this season, so a pass here is favourable to it; the 2026 test decides the final claim.

**Result: does not pass.** Δ vs plain simulator -0.073 [-0.182, +0.044]; vs pit-adjusted order -0.386 [-0.642, -0.139] places.

## Position MAE by replay point

| from | simulator + in-race GP | simulator | pit-adjusted order | frozen order |
|---|---|---|---|---|
| lap 1 | – | 1.938 | 2.744 | 1.989 |
| 25% | 1.846 | 1.795 | 2.575 | 2.052 |
| 50% | 1.307 | 1.489 | 1.709 | 1.923 |
| 75% | 0.885 | 0.973 | 0.912 | 1.152 |

## Rank correlation with the actual finish

| from | simulator + in-race GP | simulator | pit-adjusted order | frozen order |
|---|---|---|---|---|
| lap 1 | – | 0.837 | 0.683 | 0.826 |
| 25% | 0.850 | 0.866 | 0.720 | 0.815 |
| 50% | 0.913 | 0.895 | 0.862 | 0.860 |
| 75% | 0.957 | 0.950 | 0.949 | 0.930 |

## All cutoffs

| model | pos_mae | spearman | coverage_80 | p_actual_winner | gap_mae_s |
|---|---|---|---|---|---|
| simulator + in-race GP | 1.346 | 0.906 | 0.891 | 0.637 | 10.536 |
| simulator | 1.419 | 0.904 | 0.873 | 0.765 | 12.634 |
| pit-adjusted order | 1.732 | 0.844 | – | – | – |
| frozen order | 1.709 | 0.868 | – | – | – |

`coverage_80`: share of finishers inside the simulated 10–90% range (≈ 0.8 if calibrated). `p_actual_winner`: simulated probability for the actual winner. `gap_mae_s`: error in the gap to the winner for lead-lap finishers. Orders have no distribution.

## Paired differences (95% race-level bootstrap)

Negative = the first model is better.

| from | vs | Δpos_mae |
|---|---|---|
| all cutoffs | simulator | -0.073 [-0.182, +0.044] |
| all cutoffs | pit-adjusted order | -0.386 [-0.642, -0.139] |
| all cutoffs | frozen order | -0.363 [-0.604, -0.128] |
| 25% | simulator | +0.051 [-0.138, +0.274] |
| 25% | pit-adjusted order | -0.729 [-1.167, -0.289] |
| 25% | frozen order | -0.206 [-0.607, +0.173] |
| 50% | simulator | -0.182 [-0.343, -0.012] |
| 50% | pit-adjusted order | -0.402 [-0.674, -0.123] |
| 50% | frozen order | -0.616 [-1.101, -0.196] |
| 75% | simulator | -0.088 [-0.207, +0.044] |
| 75% | pit-adjusted order | -0.028 [-0.232, +0.175] |
| 75% | frozen order | -0.267 [-0.628, +0.057] |
| plain simulator, all cutoffs | pit-adjusted order | -0.313 [-0.547, -0.084] |
| plain simulator, all cutoffs | frozen order | -0.290 [-0.511, -0.067] |

## From lap 1, by race

No in-race information yet, so this tests the plain simulator across circuits.

| race | simulator | pit-adjusted order | frozen order |
|---|---|---|---|
| 2025_02 | 2.000 | 0.875 | 2.000 |
| 2025_03 | 1.050 | 0.700 | 0.800 |
| 2025_04 | 2.222 | 1.889 | 1.778 |
| 2025_05 | 0.833 | 2.778 | 1.222 |
| 2025_07 | 2.750 | 3.556 | 3.222 |
| 2025_08 | 1.778 | 2.111 | 1.333 |
| 2025_09 | 1.765 | 4.353 | 1.529 |
| 2025_11 | 1.688 | 2.750 | 2.125 |
| 2025_14 | 2.132 | 4.211 | 1.263 |
| 2025_15 | 1.912 | 4.471 | 3.059 |
| 2025_16 | 2.000 | 1.778 | 1.778 |
| 2025_17 | 2.211 | 2.211 | 2.211 |
| 2025_18 | 1.800 | 2.400 | 2.200 |
| 2025_19 | 2.289 | 1.789 | 2.000 |
| 2025_20 | 2.062 | 4.625 | 1.750 |
| 2025_21 | 1.971 | 3.176 | 2.824 |
| 2025_23 | 1.719 | 1.625 | 2.000 |
| 2025_24 | 2.700 | 4.100 | 2.700 |

## Mean of the cutoffs, by race (position MAE)

| race | event | simulator + in-race GP | simulator | pit-adjusted order | gain vs plain |
|---|---|---|---|---|---|
| 2025_02 | Chinese | 1.521 | 1.458 | 0.667 | 0.062 |
| 2025_03 | Japanese | 0.533 | 0.658 | 0.400 | -0.125 |
| 2025_04 | Bahrain | 1.574 | 1.796 | 2.037 | -0.222 |
| 2025_05 | Saudi Arabian | 0.463 | 0.426 | 1.593 | 0.037 |
| 2025_07 | Emilia Romagna | 1.806 | 2.194 | 2.185 | -0.389 |
| 2025_08 | Monaco | 1.676 | 1.444 | 1.593 | 0.231 |
| 2025_09 | Spanish | 1.784 | 1.549 | 2.510 | 0.235 |
| 2025_11 | Austrian | 0.958 | 1.052 | 1.417 | -0.094 |
| 2025_14 | Hungarian | 1.658 | 1.728 | 1.579 | -0.070 |
| 2025_15 | Dutch | 1.098 | 1.510 | 2.784 | -0.412 |
| 2025_16 | Italian | 1.222 | 1.352 | 1.370 | -0.130 |
| 2025_17 | Azerbaijan | 1.088 | 1.561 | 1.684 | -0.474 |
| 2025_18 | Singapore | 1.967 | 1.733 | 1.967 | 0.233 |
| 2025_19 | United States | 1.123 | 1.298 | 1.263 | -0.175 |
| 2025_20 | Mexico City | 1.312 | 1.396 | 1.792 | -0.083 |
| 2025_21 | São Paulo | 2.020 | 1.529 | 2.157 | 0.490 |
| 2025_23 | Qatar | 0.771 | 1.031 | 1.250 | -0.260 |
| 2025_24 | Abu Dhabi | 1.658 | 1.825 | 2.933 | -0.167 |
