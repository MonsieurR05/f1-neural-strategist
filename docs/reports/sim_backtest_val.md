# Race simulator backtest: Sakhir

_Generated 2026-10-25 19:24 UTC from commit `ef25612` by `python -m strategist.sim.backtest`. Do not edit by hand._

**Setup.** Each race is replayed from the actual order and gaps after lap 1 with the strategies, safety-car/VSC laps and retirements that actually happened, 500 Monte Carlo runs. Lap-time network and simulator parameters are fitted on the training seasons only; training-season races are in-sample, the evaluation season (val) is held out. Actual order is road order at the line (official results include post-race penalties).

**Metrics.** `pos_mae`: mean |median simulated − actual| finishing position. `spearman`: rank correlation of mean simulated and actual finishing order. `coverage_80`: share of drivers whose actual position falls in the simulated 10–90% range (≈0.8 if calibrated). `p_actual_winner`: simulated probability for the actual winner. `gap_mae_s`: error in gap to the winner for lead-lap finishers. `lap_pos_mae`: position error averaged over every lap of the race. Baselines have no distribution, so only position metrics apply.

## Summary

| role | model | pos_mae | spearman | coverage_80 | p_actual_winner | gap_mae_s | lap_pos_mae |
|---|---|---|---|---|---|---|---|
| held out | simulator | 2.125 | 0.899 | 0.850 | 0.504 | 14.891 | 1.754 |
| held out | no interactions | 2.350 | 0.899 | 0.900 | 0.478 | 15.361 | 1.780 |
| held out | lap-1 order | 2.300 | 0.856 | – | – | – | – |
| held out | grid order | 2.300 | 0.886 | – | – | – | – |
| in-sample | simulator | 1.507 | 0.898 | 0.901 | 0.422 | 21.132 | 1.722 |
| in-sample | no interactions | 1.507 | 0.882 | 0.901 | 0.397 | 21.041 | 1.782 |
| in-sample | lap-1 order | 2.201 | 0.786 | – | – | – | – |
| in-sample | grid order | 2.027 | 0.835 | – | – | – | – |

## Simulator per race

| race | role | pos_mae | spearman | coverage_80 | p_actual_winner | gap_mae_s | lap_pos_mae |
|---|---|---|---|---|---|---|---|
| 2021_01 | in-sample | 1.375 | 0.935 | 0.938 | 0.088 | 26.320 | 1.734 |
| 2022_01 | in-sample | 1.618 | 0.887 | 0.882 | 0.504 | 11.760 | 1.834 |
| 2023_01 | in-sample | 1.529 | 0.873 | 0.882 | 0.674 | 25.318 | 1.597 |
| 2024_01 | held out | 2.125 | 0.899 | 0.850 | 0.504 | 14.891 | 1.754 |

## Held-out race: Bahrain Grand Prix 2024

![simulated vs actual](..\figures\sim_backtest_2024_01.png)

| driver | actual_pos | median_pos | p10_pos | p90_pos | p_win | p_podium | actual_gap_s | median_gap_s |
|---|---|---|---|---|---|---|---|---|
| VER | 1 | 1.00 | 1.00 | 4.00 | 0.50 | 0.87 | 0.00 | 0.00 |
| PER | 2 | 6.00 | 3.00 | 10.00 | 0.02 | 0.18 | 22.46 | 54.28 |
| SAI | 3 | 5.00 | 2.00 | 8.00 | 0.03 | 0.29 | 24.98 | 49.09 |
| LEC | 4 | 2.00 | 1.00 | 5.00 | 0.34 | 0.80 | 39.60 | 14.68 |
| RUS | 5 | 4.00 | 2.00 | 7.00 | 0.07 | 0.43 | 47.02 | 41.82 |
| NOR | 6 | 7.00 | 3.00 | 11.00 | 0.01 | 0.12 | 48.38 | 61.85 |
| HAM | 7 | 8.00 | 4.00 | 12.00 | 0.00 | 0.06 | 50.26 | 70.88 |
| PIA | 8 | 8.00 | 4.00 | 11.10 | 0.01 | 0.07 | 56.05 | 67.05 |
| ALO | 9 | 6.50 | 3.00 | 10.00 | 0.01 | 0.15 | 74.80 | 60.41 |
| STR | 10 | 13.00 | 8.00 | 17.00 | 0.00 | 0.01 | 93.15 | 96.54 |
| ZHO | 11 | 16.00 | 12.00 | 19.00 | 0.00 | 0.00 | – | 113.13 |
| MAG | 12 | 15.00 | 10.00 | 18.00 | 0.00 | 0.00 | – | 105.23 |
| RIC | 13 | 15.00 | 11.00 | 18.00 | 0.00 | 0.00 | – | 108.83 |
| TSU | 14 | 11.00 | 7.00 | 15.00 | 0.00 | 0.00 | – | 88.41 |
| ALB | 15 | 12.00 | 9.00 | 17.00 | 0.00 | 0.01 | – | 93.06 |
| HUL | 16 | 11.00 | 6.90 | 15.00 | 0.00 | 0.01 | – | 84.94 |
| OCO | 17 | 16.00 | 12.00 | 19.00 | 0.00 | 0.00 | – | 116.35 |
| GAS | 18 | 19.00 | 16.00 | 20.00 | 0.00 | 0.00 | – | 136.30 |
| BOT | 19 | 17.00 | 13.00 | 19.00 | 0.00 | 0.00 | – | 117.89 |
| SAR | 20 | 19.00 | 15.00 | 20.00 | 0.00 | 0.00 | – | 129.87 |

## Calibrated simulator parameters (training seasons)

- Pit loss at Sakhir: 24.7 s (global 22.9 s, stop-to-stop σ 2.9 s); under SC/VSC ×0.69 relative to the field.
- Safety-car lap 1.55× race pace, VSC 1.35×.
- Overtaking (pairs < 1 s apart): intercept +2.20 at Sakhir (global +1.28), +0.96 per s/lap pace advantage, -5.60 per s of gap; held cars follow at 0.41–1.43 s.
- Lap-time variance split (from lap_net's out-of-season residuals): 68% race-wide shift shared by the field, 11% race-long per-driver offset, 21% lap-to-lap noise.
