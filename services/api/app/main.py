"""FastAPI app: historical races, race simulation and strategy optimisation.

Run from the repository root:

    uvicorn app.main:app --app-dir services/api --reload

Configuration: ``configs/api.yaml`` (override with ``F1S_API_CONFIG``). Model artifacts are
built by ``python -m strategist.sim.backtest``.
"""

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    DriverHistory,
    FinishRow,
    LapBands,
    LapEvent,
    OptimiseRequest,
    OptimiseResponse,
    RaceDetail,
    RaceSummary,
    SimulateRequest,
    SimulateResponse,
    StintModel,
    StrategyResult,
)
from app.state import MODEL_CAVEATS, AppState, config_path, finite_or_none
from strategist.config import load_config
from strategist.optimise.search import SearchConfig, optimise, split_at
from strategist.sim.history import strategy_from_laps
from strategist.sim.race import RaceSetup, Simulator, Stint, Strategy


def create_app(config: str | None = None) -> FastAPI:
    state: dict[str, AppState] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["s"] = AppState(config or config_path())
        yield

    app = FastAPI(title="F1 Neural Race Strategist API", version="0.1.0", lifespan=lifespan)
    origins = load_config(config or config_path()).get("cors_origins", [])
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                       allow_headers=["*"])

    def st() -> AppState:
        return state["s"]

    def race_or_404(race_id: str):
        if race_id not in st().races.index:
            raise HTTPException(404, f"unknown race {race_id!r}")
        return st().races.loc[race_id]

    def summary(race_id: str) -> RaceSummary:
        r = st().races.loc[race_id]
        return RaceSummary(race_id=race_id, year=int(r["year"]), round=int(r["round"]),
                           event_name=r["event_name"], circuit=r["circuit"],
                           total_laps=int(r["total_laps"]), wet=bool(r["wet"]),
                           simulatable=st().simulatable(race_id))

    def require_simulatable(race_id: str) -> None:
        race_or_404(race_id)
        if not st().simulatable(race_id):
            raise HTTPException(422, f"{race_id} is a wet race or had unrepresentative qualifying; "
                                     "only dry races with representative qualifying can be simulated")

    def stints_out(strategy: Strategy) -> list[StintModel]:
        return [StintModel(compound=s.compound, end_lap=s.end_lap, age_at_start=s.age_at_start)
                for s in strategy.stints]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": st().cfg["model_tag"], "races": len(st().races)}

    @app.get("/races", response_model=list[RaceSummary])
    def races(circuit: str | None = Query(None), year: int | None = Query(None)) -> list[RaceSummary]:
        r = st().races
        if circuit:
            r = r[r["circuit"].str.lower() == circuit.lower()]
        if year:
            r = r[r["year"] == year]
        return [summary(rid) for rid in r.sort_values(["year", "round"]).index]

    @app.get("/races/{race_id}", response_model=RaceDetail)
    def race(race_id: str) -> RaceDetail:
        race_or_404(race_id)
        laps = st().race_laps(race_id)
        setup, outcome = st().setup(race_id, 1) if st().simulatable(race_id) else (None, None)
        drivers = []
        grid = laps.groupby("driver")["grid_position"].first()
        teams = laps.groupby("driver")["team"].first()
        last = laps.dropna(subset=["session_time_s"]).groupby("driver").agg(
            n=("lap_number", "max"), t=("session_time_s", "last"))
        finished = set(outcome.loc[outcome["finished"], "driver"]) if outcome is not None else set(last.index)
        last["finished"] = last.index.isin(finished)
        order = last.sort_values(["finished", "n", "t"], ascending=[False, False, True]).index
        for pos, d in enumerate(order, start=1):
            strategy = strategy_from_laps(laps[laps["driver"] == d])
            drivers.append(DriverHistory(
                driver=d, team=str(teams.get(d, "")),
                grid_position=None if np.isnan(grid.get(d, np.nan)) else float(grid[d]),
                finish_position=pos, finished=bool(last.loc[d, "finished"]),
                strategy=stints_out(strategy), strategy_label=strategy.describe()))
        events = [LapEvent(lap=int(lap), kind=k) for lap, k in st().events(race_id).items() if k != "GREEN"]
        pos = laps.pivot_table(index="lap_number", columns="driver", values="position")
        t = laps.pivot_table(index="lap_number", columns="driver", values="session_time_s")
        gap = t.sub(t.min(axis=1), axis=0)
        return RaceDetail(
            race=summary(race_id), drivers=drivers, events=events,
            positions_by_lap={d: [None if np.isnan(v) else int(v) for v in pos[d]] for d in order if d in pos},
            gap_to_leader_by_lap={d: finite_or_none(gap[d]) for d in order if d in gap})

    def apply_overrides(setup: RaceSetup, overrides: dict[str, list[StintModel]]) -> None:
        for driver, stints in overrides.items():
            if driver not in [c.driver for c in setup.cars]:
                raise HTTPException(422, f"{driver} is not in this race")
            car = setup.cars[setup.driver_index(driver)]
            new = Strategy([Stint(s.compound, s.end_lap, s.age_at_start) for s in stints])
            ends = [s.end_lap for s in new.stints[:-1]]
            if any(b <= a for a, b in zip(ends, ends[1:], strict=False)) or (ends and ends[-1] >= setup.total_laps):
                raise HTTPException(422, f"{driver}: stint end laps must increase and stops must come before the last lap")
            if len({s.compound for s in new.stints}) < 2:
                raise HTTPException(422, f"{driver}: a dry race needs at least two different compounds")
            past, current, _ = split_at(car.strategy, setup.start_lap)
            new_past, new_current, _ = split_at(new, setup.start_lap)
            if [(s.compound, s.end_lap) for s in new_past] != [(s.compound, s.end_lap) for s in past] \
                    or new_current.compound != current.compound:
                raise HTTPException(422, f"{driver}: stints before lap {setup.start_lap + 1} (and the tyres "
                                         "on the car) are history and can't be changed")
            car.strategy = new

    @app.post("/simulate", response_model=SimulateResponse)
    def simulate(req: SimulateRequest) -> SimulateResponse:
        require_simulatable(req.race_id)
        cfg = st().cfg["simulate"]
        runs = min(req.runs or cfg["default_runs"], cfg["max_runs"])
        setup, _ = st().setup(req.race_id, req.start_lap)
        if req.start_lap >= setup.total_laps:
            raise HTTPException(422, "start_lap must be before the last lap")
        apply_overrides(setup, req.strategies)
        if req.events == "sampled":
            setup.events, setup.retirements = None, None
        res = Simulator(st().net, st().params, runs=runs, seed=req.seed).run(setup)
        finish = res.finish_table()
        laps: dict[str, LapBands] = {}
        lap_range = range(setup.start_lap, setup.total_laps + 1)
        pos = np.stack([res.positions(lap) for lap in lap_range])  # [laps, runs, cars]
        t = res.times[setup.start_lap:]
        lead = np.where(np.isfinite(t), t, np.inf).min(axis=2, keepdims=True)
        gap = np.where(np.isfinite(t), t - lead, np.nan)
        for i, d in enumerate(res.drivers):
            g = gap[:, :, i]
            med = [np.nanmedian(row) if np.isfinite(row).any() else np.nan for row in g]
            laps[d] = LapBands(median_pos=finite_or_none(np.median(pos[:, :, i], 1)),
                               p10_pos=finite_or_none(np.quantile(pos[:, :, i], 0.1, axis=1)),
                               p90_pos=finite_or_none(np.quantile(pos[:, :, i], 0.9, axis=1)),
                               median_gap_s=finite_or_none(med))
        notes = []
        if req.events == "actual" and req.strategies:
            notes.append("Real SC/VSC laps are kept even though changed strategies might have changed them.")
        return SimulateResponse(
            race_id=req.race_id, start_lap=setup.start_lap, runs=runs, events=req.events,
            strategies={c.driver: c.strategy.describe() for c in setup.cars},
            finish=[FinishRow(**{k: (None if isinstance(v, float) and np.isnan(v) else v)
                                 for k, v in row.items()}) for row in finish.to_dict("records")],
            laps=laps, notes=notes)

    @app.post("/optimise", response_model=OptimiseResponse)
    def optimise_strategy(req: OptimiseRequest) -> OptimiseResponse:
        require_simulatable(req.race_id)
        setup, _ = st().setup(req.race_id, req.start_lap)
        if req.driver not in [c.driver for c in setup.cars]:
            raise HTTPException(422, f"{req.driver} is not in this race (or retired on lap {req.start_lap} or earlier)")
        if req.start_lap >= setup.total_laps - 1:
            raise HTTPException(422, "start_lap is too late to change strategy")
        if req.events == "sampled":
            setup.events, setup.retirements = None, None
        else:  # hindsight: real neutralisations, other drivers' real retirements
            setup.retirements = {d: lap for d, lap in (setup.retirements or {}).items() if d != req.driver}
        c = st().cfg["optimise"]
        cfg = SearchConfig(
            max_stops=c["max_stops"] if req.max_stops is None else req.max_stops,
            pit_step=c["pit_step"], min_stint=c["min_stint"],
            screen_runs=req.screen_runs or c["screen_runs"],
            refine_runs=min(req.refine_runs or c["refine_runs"], c["max_refine_runs"]),
            refine_top=c["refine_top"], seed=req.seed)
        res = optimise(st().net, st().params, setup, req.driver, st().tyre_limits(setup.circuit), cfg)
        rows = []
        for rank, r in enumerate(res.table.head(req.top).to_dict("records"), start=1):
            rows.append(StrategyResult(
                rank=rank, strategy=r["strategy"], stints=[StintModel(**s) for s in r["stints"]],
                stops=int(r["stops"]), as_raced=bool(r["as_raced"]),
                **{k: float(r[k]) for k in ("expected_pos", "p10_pos", "p90_pos", "p_win",
                                            "p_podium", "p_points", "pit_stops_s", "tyre_wear_s",
                                            "compound_pace_s", "traffic_s")},
                time_vs_as_raced_s=finite_or_none([r["time_vs_as_raced_s"]])[0],
                p_ahead_of_as_raced=finite_or_none([r["p_ahead_of_as_raced"]])[0],
                extrapolated=bool(r["extrapolated"]), explanation=r["explanation"]))
        raced = res.table[res.table["as_raced"]]
        notes = list(res.notes)
        if not raced.empty and int(raced.index[0]) >= req.top:
            notes.append(f"The as-raced strategy ranks {int(raced.index[0]) + 1} of {len(res.table)} refined strategies.")
        return OptimiseResponse(race_id=req.race_id, driver=req.driver, start_lap=req.start_lap,
                                events=req.events, candidates_evaluated=res.candidates,
                                as_raced=res.as_raced, strategies=rows, notes=notes,
                                caveats=MODEL_CAVEATS)

    return app


app = create_app()
