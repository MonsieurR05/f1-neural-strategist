# In-race adaptation: val season (2025)

_Generated 2026-10-26 11:45 UTC from commit `dc76053` by `python -m strategist.eval.inrace`. Do not edit by hand._

**Task.** At a cutoff (a share of race distance), forecast every clean lap still to come from the clean laps run so far, given the strategy each driver actually ran. Every model starts from lap_net's pre-race forecast (fitted on 2021, 2022, 2023, 2024) and predicts its error, so `lap_net` is the no-update baseline. Laps after the cutoff are forecast as if no further safety car comes (restarts after it are not known yet). In-race models are trained on lap_net's out-of-season residuals for the same seasons (each season predicted by a lap_net fitted on the others). Scored on 80 race-cutoff cases across 20 races, 48,587 forecast laps (a lap counts once per cutoff it lies after).

**Models.** `driver's mean error so far`: shift each driver's forecast by their mean lap_net error up to the cutoff (an analyst's correction; no uncertainty). `in-race GP`: Gaussian process with race, restart, trend, drift, compound, driver and stint terms (exact posterior). `in-race GRU`: recurrent network over each driver's race so far (56 epochs, early stopping on the last training season).

## MAE (s)

|  | lap_net (pre-race, no update) | driver's mean error so far | in-race GP | in-race GRU | laps |
|---|---|---|---|---|---|
| after 10% | 0.895 | 0.931 | 0.808 | 0.716 | 18,315 |
| after 25% | 0.906 | 0.641 | 0.649 | 0.641 | 15,221 |
| after 50% | 0.918 | 0.631 | 0.630 | 0.632 | 10,088 |
| after 75% | 0.923 | 0.652 | 0.571 | 0.615 | 4,963 |
| all cutoffs | 0.906 | 0.749 | 0.697 | 0.665 | 48,587 |

## Gaussian NLL

|  | lap_net (pre-race, no update) | driver's mean error so far | in-race GP | in-race GRU | laps |
|---|---|---|---|---|---|
| after 10% | 1.616 | – | 1.497 | 1.300 | 18,315 |
| after 25% | 1.635 | – | 1.326 | 1.186 | 15,221 |
| after 50% | 1.664 | – | 1.291 | 1.195 | 10,088 |
| after 75% | 1.688 | – | 1.206 | 1.179 | 4,963 |
| all cutoffs | 1.639 | – | 1.371 | 1.230 | 48,587 |

## 90% interval coverage

Well calibrated ≈ 0.90.

|  | lap_net (pre-race, no update) | driver's mean error so far | in-race GP | in-race GRU | laps |
|---|---|---|---|---|---|
| after 10% | 0.864 | – | 0.935 | 0.920 | 18,315 |
| after 25% | 0.863 | – | 0.939 | 0.937 | 15,221 |
| after 50% | 0.865 | – | 0.919 | 0.944 | 10,088 |
| after 75% | 0.872 | – | 0.894 | 0.942 | 4,963 |
| all cutoffs | 0.865 | – | 0.929 | 0.933 | 48,587 |

### Differences vs `lap_net` (95% race-level paired bootstrap, all cutoffs)

| model | Δmae_s | Δnll |
|---|---|---|
| in_race_gp | -0.209 [-0.442, -0.008] | -0.268 [-0.590, +0.048] |
| in_race_gru | -0.241 [-0.437, -0.058] | -0.410 [-0.686, -0.185] |

### GRU vs GP

| model | Δmae_s | Δnll |
|---|---|---|
| in_race_gru | -0.032 [-0.116, +0.072] | -0.141 [-0.314, -0.021] |

## By stint (MAE, s, all cutoffs)

Laps of the stint a driver is on at the cutoff, and laps after a stop still to come (new tyres: only what the field has shown about that compound helps).

|  | lap_net (pre-race, no update) | driver's mean error so far | in-race GP | in-race GRU | laps |
|---|---|---|---|---|---|
| stint in progress at the cutoff | 0.981 | 0.738 | 0.674 | 0.611 | 19,867 |
| after a pit stop still to come | 0.854 | 0.757 | 0.712 | 0.702 | 28,720 |

## By horizon (MAE, s, all cutoffs)

|  | lap_net (pre-race, no update) | driver's mean error so far | in-race GP | in-race GRU | laps |
|---|---|---|---|---|---|
| 1–5 laps ahead | 0.884 | 0.584 | 0.434 | 0.466 | 6,518 |
| 6–15 | 0.902 | 0.671 | 0.620 | 0.593 | 13,119 |
| 16–40 | 0.912 | 0.778 | 0.746 | 0.704 | 22,021 |
| >40 | 0.915 | 0.963 | 0.931 | 0.863 | 6,929 |

## Rest-of-stint wear rate

Error in the slope of lap time against tyre age over the remaining laps (≥ 8) of stints in progress at the cutoff, s/lap. `driver's mean error so far` shifts but doesn't tilt the forecast, so it matches lap_net here.

| model | slope_mae_s_per_lap | stints |
|---|---|---|
| lap_net (pre-race, no update) | 0.050 | 1009 |
| driver's mean error so far | 0.050 | 1009 |
| in-race GP | 0.053 | 1009 |
| in-race GRU | 0.050 | 1009 |

## Fitted GP terms

Standard deviations of each source of lap_net's out-of-season error, in seconds at a 90 s lap (fitted in percent of the qualifying lap).

| term | sd_s | note |
|---|---|---|
| race (whole field) | 1.416 |  |
| after a start or restart | 1.133 | mean +0.95 s on the first green lap, decay length 0.9 laps |
| race trend | 0.833 | per 50 laps |
| race drift | 0.397 | correlation length 12 laps |
| compound in this race | 0.255 |  |
| compound wear | 0.501 | per 10 laps of tyre age |
| driver in this race | 0.529 |  |
| driver pace swings | 0.501 | correlation length 3.3 laps |
| stint | 0.040 |  |
| stint wear | 0.123 | per 10 laps of tyre age |
| lap noise | 0.475 |  |
| mean wear error | 0.236 | mean, per 10 laps (not an sd) |
