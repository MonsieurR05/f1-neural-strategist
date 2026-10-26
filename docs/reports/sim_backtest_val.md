# Race simulator backtest: Sakhir

_Generated 2026-10-25 23:23 UTC from commit `85c55ef` by `python -m strategist.sim.backtest`. Do not edit by hand._

**Setup.** Each race is replayed from the actual order and gaps after lap 1 with the strategies, safety-car/VSC laps and retirements that actually happened, 500 Monte Carlo runs. Lap-time network and simulator parameters are fitted on 2021–2023 only; races from those seasons are in-sample, the evaluation season (val) is held out. Actual order is road order at the line (official results include post-race penalties).

**Metrics.** `pos_mae`: mean |median simulated − actual| finishing position. `spearman`: rank correlation of mean simulated and actual finishing order. `coverage_80`: share of drivers whose actual position falls in the simulated 10–90% range (≈0.8 if calibrated). `p_actual_winner`: simulated probability for the actual winner. `gap_mae_s`: error in gap to the winner for lead-lap finishers. `lap_pos_mae`: position error averaged over every lap of the race. Baselines have no distribution, so only position metrics apply.

## Summary

| role | model | pos_mae | spearman | coverage_80 | p_actual_winner | gap_mae_s | lap_pos_mae |
|---|---|---|---|---|---|---|---|
| held out | simulator | 2.100 | 0.884 | 0.900 | 0.520 | 22.366 | 1.817 |
| held out | no interactions | 2.100 | 0.875 | 0.900 | 0.466 | 24.173 | 1.849 |
| held out | lap-1 order | 2.300 | 0.856 | – | – | – | – |
| held out | grid order | 2.300 | 0.886 | – | – | – | – |
| in-sample | simulator | 1.659 | 0.874 | 0.961 | 0.518 | 24.064 | 1.691 |
| in-sample | no interactions | 1.697 | 0.860 | 0.961 | 0.463 | 25.194 | 1.761 |
| in-sample | lap-1 order | 2.201 | 0.786 | – | – | – | – |
| in-sample | grid order | 2.027 | 0.835 | – | – | – | – |

## Simulator per race

| race | role | pos_mae | spearman | coverage_80 | p_actual_winner | gap_mae_s | lap_pos_mae |
|---|---|---|---|---|---|---|---|
| 2021_01 | in-sample | 1.625 | 0.897 | 1.000 | 0.200 | 24.668 | 1.814 |
| 2022_01 | in-sample | 1.824 | 0.870 | 1.000 | 0.784 | 11.735 | 1.674 |
| 2023_01 | in-sample | 1.529 | 0.855 | 0.882 | 0.570 | 35.789 | 1.583 |
| 2024_01 | held out | 2.100 | 0.884 | 0.900 | 0.520 | 22.366 | 1.817 |

## Held-out race: Bahrain Grand Prix 2024

![simulated vs actual](../figures/sim_backtest_2024_01.png)

| driver | actual_pos | median_pos | p10_pos | p90_pos | p_win | p_podium | actual_gap_s | median_gap_s |
|---|---|---|---|---|---|---|---|---|
| VER | 1 | 1.00 | 1.00 | 4.00 | 0.52 | 0.87 | 0.00 | 0.00 |
| PER | 2 | 5.00 | 2.00 | 9.00 | 0.05 | 0.33 | 22.46 | 59.14 |
| SAI | 3 | 5.00 | 2.00 | 8.00 | 0.04 | 0.28 | 24.98 | 61.08 |
| LEC | 4 | 2.00 | 1.00 | 6.00 | 0.26 | 0.70 | 39.60 | 32.20 |
| RUS | 5 | 4.00 | 2.00 | 8.00 | 0.09 | 0.41 | 47.02 | 53.47 |
| NOR | 6 | 7.00 | 3.00 | 11.00 | 0.01 | 0.12 | 48.38 | 78.67 |
| HAM | 7 | 9.00 | 5.00 | 13.00 | 0.00 | 0.04 | 50.26 | 92.34 |
| PIA | 8 | 8.00 | 5.00 | 13.00 | 0.01 | 0.04 | 56.05 | 88.13 |
| ALO | 9 | 7.00 | 3.00 | 11.00 | 0.01 | 0.16 | 74.80 | 76.50 |
| STR | 10 | 13.00 | 8.00 | 18.00 | 0.00 | 0.00 | 93.15 | 124.03 |
| ZHO | 11 | 17.00 | 12.00 | 20.00 | 0.00 | 0.00 | – | 151.79 |
| MAG | 12 | 15.00 | 10.00 | 18.00 | 0.00 | 0.00 | – | 134.40 |
| RIC | 13 | 13.00 | 9.00 | 18.00 | 0.00 | 0.00 | – | 123.21 |
| TSU | 14 | 11.00 | 7.00 | 15.00 | 0.00 | 0.01 | – | 107.65 |
| ALB | 15 | 12.00 | 8.00 | 17.00 | 0.00 | 0.01 | – | 115.96 |
| HUL | 16 | 11.00 | 6.00 | 15.00 | 0.00 | 0.02 | – | 102.16 |
| OCO | 17 | 17.00 | 13.00 | 20.00 | 0.00 | 0.00 | – | 156.72 |
| GAS | 18 | 19.00 | 14.00 | 20.00 | 0.00 | 0.00 | – | 163.87 |
| BOT | 19 | 17.00 | 12.00 | 20.00 | 0.00 | 0.00 | – | 147.81 |
| SAR | 20 | 17.00 | 13.00 | 20.00 | 0.00 | 0.00 | – | 155.81 |

## Calibrated simulator parameters (training seasons)

- Pit loss at Sakhir: 24.7 s (global 22.9 s, stop-to-stop σ 2.9 s); under SC/VSC ×0.69 relative to the field.
- Safety-car lap 1.55× race pace, VSC 1.35×.
- Overtaking (pairs < 1 s apart): intercept +2.25 at Sakhir (global +1.34), +0.92 per s/lap pace advantage, -5.66 per s of gap; held cars follow at 0.41–1.43 s.
- Lap-time variance split (from lap_net's out-of-season residuals): 62% race-wide shift shared by the field, 20% race-long per-driver offset, 18% lap-to-lap noise.
