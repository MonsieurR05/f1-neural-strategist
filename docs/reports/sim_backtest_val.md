# Race simulator backtest: Sakhir

_Generated 2026-10-25 20:03 UTC from commit `9ac83ac` by `python -m strategist.sim.backtest`. Do not edit by hand._

**Setup.** Each race is replayed from the actual order and gaps after lap 1 with the strategies, safety-car/VSC laps and retirements that actually happened, 500 Monte Carlo runs. Lap-time network and simulator parameters are fitted on the training seasons only; training-season races are in-sample, the evaluation season (val) is held out. Actual order is road order at the line (official results include post-race penalties).

**Metrics.** `pos_mae`: mean |median simulated − actual| finishing position. `spearman`: rank correlation of mean simulated and actual finishing order. `coverage_80`: share of drivers whose actual position falls in the simulated 10–90% range (≈0.8 if calibrated). `p_actual_winner`: simulated probability for the actual winner. `gap_mae_s`: error in gap to the winner for lead-lap finishers. `lap_pos_mae`: position error averaged over every lap of the race. Baselines have no distribution, so only position metrics apply.

## Summary

| role | model | pos_mae | spearman | coverage_80 | p_actual_winner | gap_mae_s | lap_pos_mae |
|---|---|---|---|---|---|---|---|
| held out | simulator | 2.200 | 0.874 | 0.950 | 0.330 | 19.793 | 1.753 |
| held out | no interactions | 2.275 | 0.863 | 0.950 | 0.306 | 20.931 | 1.848 |
| held out | lap-1 order | 2.300 | 0.856 | – | – | – | – |
| held out | grid order | 2.300 | 0.886 | – | – | – | – |
| in-sample | simulator | 1.434 | 0.881 | 0.922 | 0.514 | 22.520 | 1.712 |
| in-sample | no interactions | 1.562 | 0.898 | 0.961 | 0.454 | 23.607 | 1.791 |
| in-sample | lap-1 order | 2.201 | 0.786 | – | – | – | – |
| in-sample | grid order | 2.027 | 0.835 | – | – | – | – |

## Simulator per race

| race | role | pos_mae | spearman | coverage_80 | p_actual_winner | gap_mae_s | lap_pos_mae |
|---|---|---|---|---|---|---|---|
| 2021_01 | in-sample | 1.125 | 0.941 | 1.000 | 0.180 | 27.892 | 1.642 |
| 2022_01 | in-sample | 1.588 | 0.875 | 0.882 | 0.814 | 11.549 | 1.857 |
| 2023_01 | in-sample | 1.588 | 0.826 | 0.882 | 0.548 | 28.120 | 1.637 |
| 2024_01 | held out | 2.200 | 0.874 | 0.950 | 0.330 | 19.793 | 1.753 |

## Held-out race: Bahrain Grand Prix 2024

![simulated vs actual](..\figures\sim_backtest_2024_01.png)

| driver | actual_pos | median_pos | p10_pos | p90_pos | p_win | p_podium | actual_gap_s | median_gap_s |
|---|---|---|---|---|---|---|---|---|
| VER | 1 | 2.00 | 1.00 | 6.00 | 0.33 | 0.72 | 0.00 | 16.71 |
| PER | 2 | 7.00 | 3.00 | 12.00 | 0.02 | 0.13 | 22.46 | 65.64 |
| SAI | 3 | 5.00 | 2.00 | 9.00 | 0.05 | 0.34 | 24.98 | 48.50 |
| LEC | 4 | 2.00 | 1.00 | 5.00 | 0.40 | 0.76 | 39.60 | 13.03 |
| RUS | 5 | 4.00 | 1.00 | 8.00 | 0.13 | 0.48 | 47.02 | 42.31 |
| NOR | 6 | 7.00 | 3.00 | 11.00 | 0.02 | 0.17 | 48.38 | 65.13 |
| HAM | 7 | 8.00 | 4.00 | 13.00 | 0.01 | 0.07 | 50.26 | 75.81 |
| PIA | 8 | 7.00 | 4.00 | 12.00 | 0.01 | 0.09 | 56.05 | 72.35 |
| ALO | 9 | 6.00 | 2.00 | 11.10 | 0.03 | 0.20 | 74.80 | 62.81 |
| STR | 10 | 13.00 | 7.00 | 18.00 | 0.00 | 0.00 | 93.15 | 105.81 |
| ZHO | 11 | 16.00 | 10.00 | 20.00 | 0.00 | 0.01 | – | 121.07 |
| MAG | 12 | 14.00 | 9.00 | 18.00 | 0.00 | 0.00 | – | 113.45 |
| RIC | 13 | 15.00 | 11.00 | 19.00 | 0.00 | 0.00 | – | 123.59 |
| TSU | 14 | 11.00 | 7.00 | 17.00 | 0.00 | 0.01 | – | 97.54 |
| ALB | 15 | 13.00 | 7.00 | 18.00 | 0.00 | 0.01 | – | 105.96 |
| HUL | 16 | 12.00 | 6.00 | 17.00 | 0.00 | 0.01 | – | 95.87 |
| OCO | 17 | 15.00 | 10.00 | 20.00 | 0.00 | 0.00 | – | 120.97 |
| GAS | 18 | 18.00 | 14.00 | 20.00 | 0.00 | 0.00 | – | 138.46 |
| BOT | 19 | 17.00 | 11.00 | 20.00 | 0.00 | 0.00 | – | 127.72 |
| SAR | 20 | 18.00 | 13.00 | 20.00 | 0.00 | 0.00 | – | 132.85 |

## Calibrated simulator parameters (training seasons)

- Pit loss at Sakhir: 24.7 s (global 22.9 s, stop-to-stop σ 2.9 s); under SC/VSC ×0.69 relative to the field.
- Safety-car lap 1.55× race pace, VSC 1.35×.
- Overtaking (pairs < 1 s apart): intercept +2.25 at Sakhir (global +1.34), +0.93 per s/lap pace advantage, -5.66 per s of gap; held cars follow at 0.41–1.43 s.
- Lap-time variance split (from lap_net's out-of-season residuals): 62% race-wide shift shared by the field, 20% race-long per-driver offset, 18% lap-to-lap noise.
