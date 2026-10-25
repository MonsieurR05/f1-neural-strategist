"""Simulator parameters, calibrated from training seasons only.

Everything the race simulator needs besides the lap-time network: pit loss, safety-car and
VSC pace, the overtaking model, how a car's pace error splits into a race-long offset and
lap-to-lap noise, and (for pre-race forecasts) neutralisation and retirement hazards and a
standing-start model. ``calibrate`` writes them to JSON so the simulator and API load one
small file.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from strategist.models.features import prepare

NEUTRAL_CODES = "4567"


@dataclass
class SimParams:
    # pit stops (seconds lost relative to staying out, in/out laps combined)
    pit_loss: dict[str, float]
    pit_loss_global: float
    pit_loss_sd: float
    pit_loss_neutral_ratio: float  # under SC/VSC, relative to the field (which is slow too)
    # neutralised laps
    sc_lap_ratio: float
    vsc_lap_ratio: float
    sc_queue_gap: float
    sc_compress: float
    # overtaking: P(pass) = sigmoid(a_circuit + b·pace_advantage + c·gap) for pairs < 1 s apart
    overtake_intercept: dict[str, float]
    overtake_intercept_global: float
    overtake_pace: float
    overtake_gap: float
    held_gap_low: float
    held_gap_high: float
    # pace randomness: shares of lap-time variance that are race-long. "common" moves the whole
    # field together (track conditions; changes times, not order); "driver" is each driver's
    # race pace relative to the field. The rest is lap-to-lap noise.
    common_var_share: float
    driver_var_share: float
    # pre-race forecasts
    sc_hazard: dict[str, float] = field(default_factory=dict)
    sc_hazard_global: float = 0.0
    vsc_hazard_global: float = 0.0
    sc_durations: list[int] = field(default_factory=list)
    vsc_durations: list[int] = field(default_factory=list)
    dnf_hazard: float = 0.0
    lap1_ratio: dict[str, float] = field(default_factory=dict)
    lap1_ratio_global: float = 1.1
    grid_spacing_s: float = 0.25
    start_noise_s: float = 0.6

    def pit(self, circuit: str) -> float:
        return self.pit_loss.get(circuit, self.pit_loss_global)

    def overtake_a(self, circuit: str) -> float:
        return self.overtake_intercept.get(circuit, self.overtake_intercept_global)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "SimParams":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def is_neutral(status: pd.Series) -> pd.Series:
    return status.fillna("").astype(str).str.contains(f"[{NEUTRAL_CODES}]")


def _shrunk(counts: pd.Series, exposure: pd.Series, prior_rate: float, prior_n: float) -> dict:
    return ((counts + prior_rate * prior_n) / (exposure + prior_n)).to_dict()


def _classify(codes: str) -> str:
    if "5" in codes:
        return "RED"
    if "4" in codes:
        return "SC"
    if "6" in codes or "7" in codes:
        return "VSC"
    return "GREEN"


def lap_status(laps: pd.DataFrame) -> pd.Series:
    """Race-wide status per (race, lap number): SC, VSC, RED or GREEN, as seen by the race
    leader on that lap. Lapped cars run a given lap number later in time (a lapped car's lap
    44 happens while the leader is on lap 46), so "any car saw a code on lap n" would start
    neutralisations early for the leaders. Laps without a known leader use the majority."""
    codes = laps["track_status"].fillna("").astype(str)
    leader = laps[laps["position"] == 1].assign(codes=codes).drop_duplicates(["race_id", "lap_number"])
    by_leader = leader.set_index(["race_id", "lap_number"])["codes"].map(_classify)
    majority = (laps.assign(kind=codes.map(_classify))
                .groupby(["race_id", "lap_number"])["kind"].agg(lambda s: s.mode().iat[0]))
    return by_leader.reindex(majority.index).fillna(majority)


def _pit_losses(laps: pd.DataFrame) -> pd.DataFrame:
    """Two-lap time (in-lap + out-lap) of each stop minus what the driver would have done."""
    laps = laps.sort_values(["race_id", "driver", "lap_number"]).copy()
    ref = laps[laps["exclude_reason"].isna()].groupby(["race_id", "driver"])["lap_time_s"].median()
    laps = laps.join(ref.rename("ref"), on=["race_id", "driver"])
    g = laps.groupby(["race_id", "driver"])
    laps["two_laps"] = g["session_time_s"].shift(-1) - g["session_time_s"].shift(1)
    laps["neutral"] = is_neutral(laps["track_status"]) | is_neutral(g["track_status"].shift(-1))
    # field reference for the same two laps: median over cars not stopping on them
    stopping = laps["pit_in_time_s"].notna() | laps["pit_out_time_s"].notna()
    field = laps[~stopping].groupby(["race_id", "lap_number"])["two_laps"].median()
    laps = laps.join(field.rename("field_two_laps"), on=["race_id", "lap_number"])
    stops = laps[laps["pit_in_time_s"].notna() & laps["lap_number"].gt(1)].copy()
    stops["loss_green"] = stops["two_laps"] - 2 * stops["ref"]
    stops["loss_field"] = stops["two_laps"] - stops["field_two_laps"]
    return stops


def _overtake_pairs(laps: pd.DataFrame, mu: pd.Series) -> pd.DataFrame:
    """Pairs < 1 s apart at the start of a green lap, with the follower's pace advantage."""
    laps = laps.assign(mu=mu).dropna(subset=["session_time_s"])
    rows = []
    for _, g in laps.groupby("race_id"):
        by_lap = {n: x.sort_values("session_time_s").set_index("driver") for n, x in g.groupby("lap_number")}
        for n, end in by_lap.items():
            if n < 3 or n - 1 not in by_lap:
                continue
            start = by_lap[n - 1]
            order = [d for d in start.index if d in end.index]
            for a, b in zip(order[:-1], order[1:], strict=True):
                gap = start.at[b, "session_time_s"] - start.at[a, "session_time_s"]
                ea, eb = end.loc[a], end.loc[b]
                if gap >= 1.0 or pd.isna(ea["mu"]) or pd.isna(eb["mu"]):
                    continue
                if (pd.notna(ea["pit_in_time_s"]) or pd.notna(eb["pit_in_time_s"])
                        or pd.notna(ea["pit_out_time_s"]) or pd.notna(eb["pit_out_time_s"])
                        or any(c in str(ea["track_status"]) + str(eb["track_status"])
                               for c in NEUTRAL_CODES)):
                    continue
                rows.append({"circuit": ea["circuit"], "gap": gap, "pace": ea["mu"] - eb["mu"],
                             "passed": eb["session_time_s"] < ea["session_time_s"],
                             "gap_end": eb["session_time_s"] - ea["session_time_s"]})
    return pd.DataFrame(rows)


def _neutral_periods(status: pd.Series) -> pd.DataFrame:
    rows = []
    for race_id, s in status.groupby(level=0):
        s = s.droplevel(0).sort_index()
        run_kind, run_len = None, 0
        for kind in [*s.to_list(), "END"]:
            k = "SC" if kind == "RED" else kind
            if k == run_kind:
                run_len += 1
                continue
            if run_kind in ("SC", "VSC"):
                rows.append({"race_id": race_id, "kind": run_kind, "laps": run_len})
            run_kind, run_len = k, 1
    return pd.DataFrame(rows, columns=["race_id", "kind", "laps"])


def calibrate(laps: pd.DataFrame, races: pd.DataFrame, net, seasons: list[int]) -> SimParams:
    """Fit simulator parameters on dry races from ``seasons`` with the trained lap network."""
    dry = races[~races["wet"] & races["year"].isin(seasons)]
    laps = laps[laps["race_id"].isin(dry["race_id"])]

    stops = _pit_losses(laps)
    green = stops[~stops["neutral"]]["loss_green"].dropna()
    green = green[green.between(10, 45)]
    by_circuit = stops[~stops["neutral"]].assign(loss=stops["loss_green"]).dropna(subset=["loss"])
    by_circuit = by_circuit[by_circuit["loss"].between(10, 45)].groupby("circuit")["loss"]
    pit_loss = by_circuit.median()[by_circuit.size() >= 10].round(2).to_dict()
    neutral = stops[stops["neutral"]]["loss_field"].dropna()
    neutral_ratio = float((neutral[neutral.between(0, 45)] / green.median()).median())

    ref = laps[laps["exclude_reason"].isna()].groupby(["race_id", "driver"])["lap_time_s"].median()
    status = lap_status(laps)
    tagged = laps.join(status.rename("race_status"), on=["race_id", "lap_number"])
    tagged = tagged.join(ref.rename("ref"), on=["race_id", "driver"])
    ratio = tagged["lap_time_s"] / tagged["ref"]
    pure = tagged["track_status"].astype(str)
    sc_ratio = float(ratio[pure.eq("4")].median())
    vsc_ratio = float(ratio[pure.isin(["6", "67", "7"])].median())

    prepared = prepare(laps.dropna(subset=["tyre_life", "lap_time_s"]))
    mu, _ = net.predict_dist(prepared)
    pairs = _overtake_pairs(laps, mu.reindex(laps.index))
    circuits = sorted(pairs["circuit"].unique())
    X = np.column_stack([pairs["pace"], pairs["gap"],
                         pd.get_dummies(pd.Categorical(pairs["circuit"], circuits), dtype=float)])
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, pairs["passed"])
    coef, b0 = clf.coef_[0], float(clf.intercept_[0])
    held = pairs.loc[~pairs["passed"], "gap_end"]

    res = net.oos_residuals
    race_mean = res.groupby("race_id")["resid_pct"].transform("mean")
    driver_mean = res.groupby(["race_id", "driver"])["resid_pct"].transform("mean")
    total_var = float(np.var(res["resid_pct"]))
    common_share = float(np.var(race_mean) / total_var)
    driver_share = float(np.var(driver_mean - race_mean) / total_var)

    periods = _neutral_periods(status)
    race_laps = status.groupby(level=0).size()
    circuit_of = dry.set_index("race_id")["circuit"]
    sc_counts = periods[periods["kind"] == "SC"].groupby("race_id").size().reindex(race_laps.index, fill_value=0)
    sc_global = float(sc_counts.sum() / race_laps.sum())
    vsc_global = float((periods["kind"] == "VSC").sum() / race_laps.sum())
    sc_hazard = _shrunk(sc_counts.groupby(circuit_of).sum(), race_laps.groupby(circuit_of).sum(),
                        sc_global, prior_n=150)

    last_lap = laps.groupby(["race_id", "driver"])["lap_number"].max()
    race_len = laps.groupby("race_id")["lap_number"].max()
    dnf = (last_lap < race_len.reindex(last_lap.index.get_level_values(0)).to_numpy() - 3).sum()

    lap1 = laps[laps["lap_number"] == 1].dropna(subset=["session_time_s", "grid_position"])
    lap1 = lap1[lap1["grid_position"] > 0]
    lap1 = lap1.assign(gap=lap1["session_time_s"] - lap1.groupby("race_id")["session_time_s"].transform("min"))
    slope, _ = np.polyfit(lap1["grid_position"], lap1["gap"], 1)
    resid = lap1["gap"] - slope * lap1["grid_position"]
    lead = laps[(laps["lap_number"] == 1) & (laps["position"] == 1)].dropna(subset=["lap_time_s"])
    lap1_ratio = (lead["lap_time_s"] / lead["quali_pole_s"]).groupby(lead["circuit"]).median()

    return SimParams(
        pit_loss=pit_loss, pit_loss_global=round(float(green.median()), 2),
        pit_loss_sd=round(float(1.4826 * (green - green.median()).abs().median()), 2),
        pit_loss_neutral_ratio=round(neutral_ratio, 3),
        sc_lap_ratio=round(sc_ratio, 3), vsc_lap_ratio=round(vsc_ratio, 3),
        sc_queue_gap=0.8, sc_compress=0.35,
        overtake_intercept={c: round(b0 + float(w), 3) for c, w in zip(circuits, coef[2:], strict=True)},
        overtake_intercept_global=round(b0 + float(np.mean(coef[2:])), 3),
        overtake_pace=round(float(coef[0]), 3), overtake_gap=round(float(coef[1]), 3),
        held_gap_low=round(float(held.quantile(0.1)), 3), held_gap_high=round(float(held.quantile(0.9)), 3),
        common_var_share=round(common_share, 3), driver_var_share=round(driver_share, 3),
        sc_hazard={k: round(v, 5) for k, v in sc_hazard.items()},
        sc_hazard_global=round(sc_global, 5), vsc_hazard_global=round(vsc_global, 5),
        sc_durations=periods.loc[periods["kind"] == "SC", "laps"].astype(int).tolist(),
        vsc_durations=periods.loc[periods["kind"] == "VSC", "laps"].astype(int).tolist(),
        dnf_hazard=round(float(dnf / len(laps)), 5),
        lap1_ratio=lap1_ratio.round(3).to_dict(),
        lap1_ratio_global=round(float(lap1_ratio.median()), 3),
        grid_spacing_s=round(float(slope), 3), start_noise_s=round(float(resid.std()), 3),
    )
