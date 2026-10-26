# Race simulator backtest: Sakhir

_Generated 2026-10-25 23:23 UTC from commit `85c55ef` by `python -m strategist.sim.backtest`. Do not edit by hand._

**Setup.** Each race is replayed from the actual order and gaps after lap 1 with the strategies, safety-car/VSC laps and retirements that actually happened, 500 Monte Carlo runs. Lap-time network and simulator parameters are fitted on 2021–2024 only; races from those seasons are in-sample, the evaluation season (test) is held out. Actual order is road order at the line (official results include post-race penalties).

**Metrics.** `pos_mae`: mean |median simulated − actual| finishing position. `spearman`: rank correlation of mean simulated and actual finishing order. `coverage_80`: share of drivers whose actual position falls in the simulated 10–90% range (≈0.8 if calibrated). `p_actual_winner`: simulated probability for the actual winner. `gap_mae_s`: error in gap to the winner for lead-lap finishers. `lap_pos_mae`: position error averaged over every lap of the race. Baselines have no distribution, so only position metrics apply.

## Summary

| role | model | pos_mae | spearman | coverage_80 | p_actual_winner | gap_mae_s | lap_pos_mae |
|---|---|---|---|---|---|---|---|
| held out | simulator | 2.222 | 0.814 | 0.833 | 0.736 | 7.538 | 1.938 |
| held out | no interactions | 2.389 | 0.808 | 0.889 | 0.662 | 7.369 | 1.985 |
| held out | lap-1 order | 1.778 | 0.874 | – | – | – | – |
| held out | grid order | 2.778 | 0.775 | – | – | – | – |
| in-sample | simulator | 1.712 | 0.882 | 0.870 | 0.584 | 18.666 | 1.699 |
| in-sample | no interactions | 1.678 | 0.877 | 0.900 | 0.540 | 19.131 | 1.719 |
| in-sample | lap-1 order | 2.226 | 0.803 | – | – | – | – |
| in-sample | grid order | 2.095 | 0.848 | – | – | – | – |

## Simulator per race

| race | role | pos_mae | spearman | coverage_80 | p_actual_winner | gap_mae_s | lap_pos_mae |
|---|---|---|---|---|---|---|---|
| 2021_01 | in-sample | 1.688 | 0.918 | 0.875 | 0.108 | 22.797 | 1.872 |
| 2022_01 | in-sample | 1.588 | 0.870 | 0.882 | 0.890 | 11.794 | 1.642 |
| 2023_01 | in-sample | 1.471 | 0.846 | 0.824 | 0.718 | 26.202 | 1.483 |
| 2024_01 | in-sample | 2.100 | 0.896 | 0.900 | 0.618 | 13.869 | 1.798 |
| 2025_04 | held out | 2.222 | 0.814 | 0.833 | 0.736 | 7.538 | 1.938 |

## Held-out race: Bahrain Grand Prix 2025

![simulated vs actual](../figures/sim_backtest_2025_04.png)

| driver | actual_pos | median_pos | p10_pos | p90_pos | p_win | p_podium | actual_gap_s | median_gap_s |
|---|---|---|---|---|---|---|---|---|
| PIA | 1 | 1.00 | 1.00 | 3.00 | 0.74 | 0.97 | 0.00 | 0.00 |
| RUS | 2 | 2.00 | 1.00 | 4.00 | 0.17 | 0.78 | 15.73 | 14.45 |
| NOR | 3 | 4.00 | 2.00 | 6.00 | 0.02 | 0.37 | 16.18 | 21.34 |
| LEC | 4 | 3.00 | 2.00 | 5.00 | 0.07 | 0.67 | 19.56 | 15.57 |
| HAM | 5 | 7.00 | 5.00 | 10.00 | 0.00 | 0.02 | 27.94 | 37.46 |
| VER | 6 | 8.00 | 6.00 | 10.00 | 0.00 | 0.01 | 34.31 | 40.06 |
| GAS | 7 | 5.00 | 3.00 | 7.00 | 0.00 | 0.11 | 35.86 | 27.68 |
| OCO | 8 | 15.00 | 11.00 | 18.00 | 0.00 | 0.00 | 44.18 | 65.93 |
| TSU | 9 | 9.00 | 7.00 | 11.00 | 0.00 | 0.00 | 44.99 | 43.83 |
| BEA | 10 | 15.00 | 11.00 | 18.00 | 0.00 | 0.00 | 47.57 | 66.39 |
| ANT | 11 | 6.00 | 4.00 | 8.00 | 0.00 | 0.08 | 47.90 | 30.69 |
| ALB | 12 | 14.00 | 10.00 | 17.00 | 0.00 | 0.00 | 48.72 | 61.69 |
| LAW | 13 | 15.00 | 10.00 | 18.00 | 0.00 | 0.00 | 49.31 | 64.70 |
| DOO | 14 | 12.00 | 9.00 | 16.00 | 0.00 | 0.00 | 52.65 | 56.15 |
| HAD | 15 | 13.00 | 10.00 | 17.00 | 0.00 | 0.00 | 56.23 | 58.44 |
| ALO | 16 | 12.00 | 9.00 | 16.00 | 0.00 | 0.00 | 60.23 | 55.72 |
| STR | 17 | 16.00 | 12.00 | 18.00 | 0.00 | 0.00 | 65.33 | 67.95 |
| BOR | 18 | 16.00 | 12.00 | 18.00 | 0.00 | 0.00 | 66.83 | 68.47 |
| HUL | 19 | 19.00 | 19.00 | 19.00 | 0.00 | 0.00 | 53.31 | – |
| SAI | 20 | 20.00 | 20.00 | 20.00 | 0.00 | 0.00 | – | – |

## Calibrated simulator parameters (training seasons)

- Pit loss at Sakhir: 24.8 s (global 22.9 s, stop-to-stop σ 3.1 s); under SC/VSC ×0.68 relative to the field.
- Safety-car lap 1.55× race pace, VSC 1.35×.
- Overtaking (pairs < 1 s apart): intercept +2.19 at Sakhir (global +1.30), +1.02 per s/lap pace advantage, -5.62 per s of gap; held cars follow at 0.41–1.42 s.
- Lap-time variance split (from lap_net's out-of-season residuals): 63% race-wide shift shared by the field, 11% race-long per-driver offset, 26% lap-to-lap noise.
