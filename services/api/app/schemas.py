"""Request and response models. Every response says where its numbers come from:
``historical`` (what happened) or ``simulated`` (counterfactual estimates)."""

from typing import Literal

from pydantic import BaseModel, Field

Compound = Literal["SOFT", "MEDIUM", "HARD"]
Provenance = Literal["historical", "simulated"]
EventsMode = Literal["actual", "sampled"]


class StintModel(BaseModel):
    compound: Compound
    end_lap: int = Field(ge=1, description="last lap on this set (the in-lap if a stop follows)")
    age_at_start: int = Field(0, ge=0, description="laps already on the set when fitted")


class RaceSummary(BaseModel):
    race_id: str
    year: int
    round: int
    event_name: str
    circuit: str
    total_laps: int
    wet: bool
    simulatable: bool = Field(description="dry race with representative qualifying")


class DriverHistory(BaseModel):
    driver: str
    team: str
    grid_position: float | None
    finish_position: int = Field(description="road order at the line (official results may include penalties)")
    finished: bool
    strategy: list[StintModel]
    strategy_label: str


class LapEvent(BaseModel):
    lap: int
    kind: Literal["SC", "VSC", "RED"]


class RaceDetail(BaseModel):
    provenance: Provenance = "historical"
    race: RaceSummary
    drivers: list[DriverHistory]
    events: list[LapEvent]
    positions_by_lap: dict[str, list[int | None]]
    gap_to_leader_by_lap: dict[str, list[float | None]]
    leader_lap_time_s: list[float | None] = Field(description="race leader's time for each lap")
    nominal_lap_s: float = Field(description="median clean racing lap, for animation")


class Track(BaseModel):
    circuit: str
    source: str
    points: list[tuple[float, float]] = Field(description="normalised x, y; first point on the start/finish line")
    fraction: list[float] = Field(description="share of the lap distance at each point")


class SimulateRequest(BaseModel):
    race_id: str
    start_lap: int = Field(1, ge=1, description="replay from the actual state after this many laps")
    runs: int | None = Field(None, ge=10)
    seed: int = 0
    events: EventsMode = Field("actual", description="actual: the race's real SC/VSC laps and retirements; sampled: drawn from history")
    strategies: dict[str, list[StintModel]] = Field(default_factory=dict,
                                                    description="per-driver strategy overrides (full stint list)")


class FinishRow(BaseModel):
    driver: str
    mean_pos: float
    median_pos: float
    p10_pos: float
    p90_pos: float
    p_win: float
    p_podium: float
    p_points: float
    median_gap_s: float | None
    mean_held_s: float


class LapBands(BaseModel):
    median_pos: list[float | None]
    p10_pos: list[float | None]
    p90_pos: list[float | None]
    median_gap_s: list[float | None]


class SimulateResponse(BaseModel):
    provenance: Provenance = "simulated"
    race_id: str
    start_lap: int
    runs: int
    events: EventsMode
    strategies: dict[str, str]
    finish: list[FinishRow]
    laps: dict[str, LapBands] = Field(description="per driver, index 0 = lap start_lap")
    notes: list[str]


class OptimiseRequest(BaseModel):
    race_id: str
    driver: str
    start_lap: int = Field(1, ge=1)
    events: EventsMode = Field("sampled", description="sampled: decide under uncertainty; actual: hindsight with the real SC/VSC laps")
    max_stops: int | None = Field(None, ge=0, le=3)
    screen_runs: int | None = Field(None, ge=4)
    refine_runs: int | None = Field(None, ge=20)
    top: int = Field(10, ge=1, le=50)
    seed: int = 0


class StrategyResult(BaseModel):
    rank: int
    strategy: str
    stints: list[StintModel]
    stops: int
    as_raced: bool
    expected_pos: float
    p10_pos: float
    p90_pos: float
    p_win: float
    p_podium: float
    p_points: float
    time_vs_as_raced_s: float | None
    p_ahead_of_as_raced: float | None
    pit_stops_s: float
    tyre_wear_s: float
    compound_pace_s: float
    traffic_s: float
    extrapolated: bool
    explanation: str


class OptimiseResponse(BaseModel):
    provenance: Provenance = "simulated"
    race_id: str
    driver: str
    start_lap: int
    events: EventsMode
    candidates_evaluated: int
    as_raced: str
    strategies: list[StrategyResult]
    notes: list[str]
    caveats: list[str]
