"""Circuit outlines for the race replay: one clean racing lap of car positions (FastF1).

Usage (from the repository root):

    python -m strategist.data.tracks --config configs/data.yaml --circuit Sakhir

Writes ``<processed>/tracks/<circuit>.json``: points normalised to a unit box (aspect kept,
y pointing up as on the official maps), starting at the start/finish line, plus the
cumulative distance fraction of each point so a car's progress through the lap maps to a
position on the outline. Uses the latest race at the circuit that FastF1 can load (its
community mirror covers 2021–2022, enough for layouts unchanged since).
"""

import argparse
import json
import logging
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

from strategist.config import load_config

log = logging.getLogger(__name__)


def outline(year: int, rnd: int, lap_number: int = 10) -> dict:
    session = fastf1.get_session(year, rnd, "R")
    session.load(laps=True, telemetry=True, weather=False, messages=False)
    laps = session.laps
    winner = session.results.sort_values("Position")["Abbreviation"].iloc[0]
    lap = laps.pick_drivers(winner).pick_laps(lap_number).iloc[0]
    pos = lap.get_pos_data()[["X", "Y"]].to_numpy(float)
    pos = pos[np.r_[True, np.any(np.diff(pos, axis=0) != 0, axis=1)]]  # drop repeated samples
    xy = pos - pos.mean(0)
    xy /= np.abs(xy).max()
    step = np.hypot(*np.diff(np.vstack([xy, xy[:1]]), axis=0).T)
    frac = np.concatenate([[0.0], np.cumsum(step)[:-1]]) / step.sum()
    return {"source": f"FastF1 position data, {year} round {rnd}, {winner} lap {lap_number}",
            "points": np.round(xy, 4).tolist(), "fraction": np.round(frac, 5).tolist()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--circuit", nargs="+", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fastf1.set_log_level("WARNING")

    cfg = load_config(args.config)
    fastf1.Cache.configure(cache_dir=cfg["paths"]["fastf1_cache"])
    processed = Path(cfg["paths"]["processed"])
    races = pd.read_parquet(processed / "races.parquet")
    out_dir = processed / "tracks"
    out_dir.mkdir(parents=True, exist_ok=True)
    for circuit in args.circuit:
        options = races[(races["circuit"] == circuit) & (races["source"] == "fastf1")]
        if options.empty:
            log.error("%s: no FastF1-sourced race to take the layout from", circuit)
            continue
        race = options.sort_values("year").iloc[-1]
        track = outline(int(race["year"]), int(race["round"]))
        (out_dir / f"{circuit}.json").write_text(json.dumps(track), encoding="utf-8")
        log.info("%s: %d points from %s", circuit, len(track["points"]), track["source"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
