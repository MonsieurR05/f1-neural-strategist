import numpy as np
import pandas as pd
import pytest

from strategist.data.clean import RENAME
from strategist.data.sources import LAP_COLUMNS
from strategist.data.sources.openf1 import (
    build_laps,
    expand_stints,
    lap_track_status,
    quali_best_laps,
    track_status_events,
)

ORIGIN = pd.Timestamp("2024-03-02T15:00:00Z")


def iso(seconds: float) -> str:
    return (ORIGIN + pd.to_timedelta(float(seconds), unit="s")).isoformat()


def race_control(*rows) -> pd.DataFrame:
    cols = ["date", "category", "flag", "scope", "message"]
    return pd.DataFrame([dict(zip(cols, (iso(r[0]), *r[1:]), strict=True)) for r in rows])


# Shape of 2023 Australia: SC, then SC -> red flag, VSC, sector flags that must be ignored.
RC = race_control(
    (0, "Flag", "GREEN", "Track", "GREEN LIGHT - PIT EXIT OPEN"),
    (100, "SafetyCar", None, None, "SAFETY CAR DEPLOYED"),
    (250, "SafetyCar", None, None, "SAFETY CAR IN THIS LAP"),
    (300, "Flag", "CLEAR", "Track", "TRACK CLEAR"),
    (350, "Flag", "YELLOW", "Sector", "YELLOW IN TRACK SECTOR 4"),
    (400, "SafetyCar", None, None, "VIRTUAL SAFETY CAR DEPLOYED"),
    (450, "SafetyCar", None, None, "VIRTUAL SAFETY CAR ENDING"),
    (460, "Flag", "CLEAR", "Track", "TRACK CLEAR"),
    (600, "SafetyCar", None, None, "SAFETY CAR DEPLOYED"),
    (650, "Flag", "RED", "Track", "RED FLAG"),
)


def test_track_status_events_map_to_fastf1_codes():
    codes = [c for _, c in track_status_events(RC, ORIGIN)]
    assert codes == ["1", "4", "1", "6", "7", "1", "4", "5"]


def test_lap_track_status_includes_state_at_start_and_changes_during_lap():
    events = track_status_events(RC, ORIGIN)
    starts = np.array([10, 90, 200, 290, 380, 440, 500, 620, np.nan])
    ends = np.array([90, 190, 290, 380, 440, 500, 600, 700, np.nan])
    assert lap_track_status(starts, ends, events) == [
        "1",    # green
        "14",   # SC deployed mid-lap
        "4",    # under SC ("in this lap" does not end it)
        "41",   # track clear mid-lap
        "16",   # VSC deployed
        "671",  # VSC ending, then clear
        "14",   # SC deployed exactly at lap end
        "45",   # red flag
        "",     # no timing
    ]


def test_expand_stints_tyre_life_convention():
    stints = pd.DataFrame([
        {"driver_number": 1, "stint_number": 1, "lap_start": 1, "lap_end": 3,
         "compound": "SOFT", "tyre_age_at_start": 3},
        {"driver_number": 1, "stint_number": 2, "lap_start": 4, "lap_end": None,
         "compound": "HARD", "tyre_age_at_start": 0},
    ])
    out = expand_stints(stints, max_lap=6).set_index("LapNumber")
    assert out.loc[1.0, "TyreLife"] == 4 and not out.loc[1.0, "FreshTyre"]
    assert out.loc[3.0, "TyreLife"] == 6
    assert out.loc[4.0, "TyreLife"] == 1 and out.loc[4.0, "FreshTyre"]
    assert out.loc[6.0, "Compound"] == "HARD"  # open-ended final stint runs to max_lap


def test_quali_best_laps_skips_deleted_laps():
    laps = pd.DataFrame({"driver_number": [4, 4, 4, 81],
                         "lap_duration": [86.123, 86.400, None, 86.300]})
    rc = pd.DataFrame({"message": [
        "CAR 4 (NOR) TIME 1:26.123 DELETED - TRACK LIMITS AT TURN 17 LAP 5 15:04:11",
        "GREEN LIGHT - PIT EXIT OPEN",
    ]})
    drivers = pd.DataFrame({"driver_number": [4, 81], "name_acronym": ["NOR", "PIA"]})
    best = quali_best_laps(laps, rc, drivers).set_index("Driver")["QualiBestS"]
    assert best.to_dict() == {"NOR": 86.4, "PIA": 86.3}


@pytest.fixture
def raw_race() -> dict[str, pd.DataFrame]:
    """Two drivers, 4 laps; driver 1 pits at the end of lap 2."""
    laps, t = [], {1: 0.0, 2: 1.0}
    for lap in range(1, 5):
        for drv in (1, 2):
            dur = 100.0 if (drv, lap) not in {(1, 2), (1, 3)} else 120.0
            laps.append({"driver_number": drv, "lap_number": lap, "date_start": iso(t[drv]),
                         "lap_duration": dur, "duration_sector_1": 30.0,
                         "duration_sector_2": 40.0, "duration_sector_3": dur - 70.0,
                         "is_pit_out_lap": (drv, lap) == (1, 3)})
            t[drv] += dur
    return {
        "laps": pd.DataFrame(laps),
        "stints": pd.DataFrame([
            {"driver_number": 1, "stint_number": 1, "lap_start": 1, "lap_end": 2,
             "compound": "MEDIUM", "tyre_age_at_start": 0},
            {"driver_number": 1, "stint_number": 2, "lap_start": 3, "lap_end": 4,
             "compound": "HARD", "tyre_age_at_start": 0},
            {"driver_number": 2, "stint_number": 1, "lap_start": 1, "lap_end": 4,
             "compound": "MEDIUM", "tyre_age_at_start": 0},
        ]),
        "pit": pd.DataFrame([{"driver_number": 1, "lap_number": 2, "date": iso(215)}]),
        "race_control": RC.iloc[:1],
        "weather": pd.DataFrame([
            {"date": iso(-30), "air_temperature": 20.0, "track_temperature": 30.0,
             "humidity": 50.0, "pressure": 1010.0, "rainfall": 0, "wind_speed": 1.0},
            {"date": iso(290), "air_temperature": 21.0, "track_temperature": 31.0,
             "humidity": 50.0, "pressure": 1010.0, "rainfall": 0, "wind_speed": 1.0},
        ]),
        "drivers": pd.DataFrame([
            {"driver_number": 1, "name_acronym": "AAA", "team_name": "Team A"},
            {"driver_number": 2, "name_acronym": "BBB", "team_name": "Team B"},
        ]),
    }


def test_build_laps_schema_matches_clean_contract(raw_race):
    laps = build_laps(raw_race, ORIGIN)
    assert list(laps.columns) == LAP_COLUMNS
    assert set(RENAME) - {"Year", "Round"} <= set(laps.columns)


def test_build_laps_pits_positions_weather(raw_race):
    laps = build_laps(raw_race, ORIGIN).set_index(["Driver", "LapNumber"])
    assert laps["PitInTime"].notna().sum() == 1 and pd.notna(laps.loc[("AAA", 2.0), "PitInTime"])
    assert laps["PitOutTime"].notna().sum() == 1 and pd.notna(laps.loc[("AAA", 3.0), "PitOutTime"])
    assert laps.loc[("AAA", 3.0), ["Compound", "Stint", "TyreLife"]].tolist() == ["HARD", 2.0, 1.0]
    assert laps.loc[("AAA", 1.0), "Time"] == pytest.approx(100.0)
    assert laps.loc[("AAA", 1.0), "Position"] == 1 and laps.loc[("AAA", 3.0), "Position"] == 2
    assert laps.loc[("AAA", 1.0), "AirTemp"] == 20.0 and laps.loc[("BBB", 4.0), "AirTemp"] == 21.0
    assert laps["TrackStatus"].eq("1").all()
    assert not laps["Rainfall"].any()
