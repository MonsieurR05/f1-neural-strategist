import numpy as np
import pandas as pd
import pytest

from strategist.data.clean import (
    add_intervals,
    add_quali_prior,
    assign_exclusions,
    circuit_key,
    is_wet_race,
)


def make_laps(n_laps: int = 5, drivers=("AAA", "BBB"), base: float = 90.0) -> pd.DataFrame:
    """Two cars, AAA 1.5 s ahead of BBB at every line crossing, all laps clean by default."""
    rows = []
    for i, drv in enumerate(drivers):
        for lap in range(1, n_laps + 1):
            rows.append({
                "driver": drv,
                "lap_number": lap,
                "lap_time_s": base,
                "session_time_s": lap * base + i * 1.5,
                "compound": "MEDIUM",
                "track_status": "1",
                "pit_in_time_s": np.nan,
                "pit_out_time_s": np.nan,
                "generated": False,
                "is_accurate": True,
                "rainfall": False,
            })
    return pd.DataFrame(rows)


def exclusions(laps: pd.DataFrame, wet_race: bool = False) -> pd.Series:
    return assign_exclusions(
        laps, wet_race=wet_race, dry_only=True,
        neutralised_status=["4", "5", "6", "7"], slow_outlier_ratio=1.07,
    )


def test_clean_laps_have_no_reason_except_lap1():
    reason = exclusions(make_laps())
    laps = make_laps()
    assert (reason[laps["lap_number"] == 1] == "lap1").all()
    assert reason[laps["lap_number"] > 1].isna().all()


def test_first_matching_reason_wins():
    laps = make_laps()
    laps.loc[2, ["pit_in_time_s", "track_status"]] = [300.0, "14"]
    laps.loc[3, "track_status"] = "126"
    laps.loc[4, "compound"] = "INTERMEDIATE"
    reason = exclusions(laps)
    assert reason[2] == "pit_in"
    assert reason[3] == "neutralised"
    assert reason[4] == "non_slick"


def test_yellow_flag_alone_is_not_neutralised():
    laps = make_laps()
    laps.loc[2, "track_status"] = "12"
    assert pd.isna(exclusions(laps)[2])


def test_slow_outlier_against_clean_median():
    laps = make_laps()
    laps.loc[2, "lap_time_s"] = 90.0 * 1.10
    laps.loc[3, "lap_time_s"] = 90.0 * 1.05
    reason = exclusions(laps)
    assert reason[2] == "slow_outlier"
    assert pd.isna(reason[3])


def test_wet_race_excludes_everything():
    assert exclusions(make_laps(), wet_race=True).eq("wet_race").all()


def test_is_wet_race_uses_shares_not_any():
    laps = make_laps(n_laps=10)  # 20 laps
    laps.loc[0, "rainfall"] = True
    laps.loc[1, "compound"] = "INTERMEDIATE"
    assert not is_wet_race(laps, inter_share=0.10, rain_share=0.25)
    laps.loc[2, "compound"] = "INTERMEDIATE"
    assert is_wet_race(laps, inter_share=0.10, rain_share=0.25)
    laps["compound"] = "MEDIUM"
    laps.loc[:4, "rainfall"] = True
    assert is_wet_race(laps, inter_share=0.10, rain_share=0.25)


def test_rain_and_damp_laps_excluded_in_dry_race():
    laps = make_laps(n_laps=10)
    laps.loc[(laps.driver == "AAA") & (laps.lap_number == 2), "compound"] = "INTERMEDIATE"
    laps.loc[(laps.driver == "BBB") & (laps.lap_number == 8), "rainfall"] = True
    reason = exclusions(laps).set_axis(pd.MultiIndex.from_frame(laps[["driver", "lap_number"]]))
    assert reason[("AAA", 2)] == "non_slick"
    assert reason[("BBB", 2)] == "damp" and reason[("BBB", 4)] == "damp"
    assert pd.isna(reason[("BBB", 5)])
    assert reason[("BBB", 8)] == "rain"
    assert pd.isna(reason[("AAA", 8)])


def test_interval_start_is_previous_lap_gap():
    laps = add_intervals(make_laps()).set_index(["driver", "lap_number"])
    assert np.isnan(laps.loc[("AAA", 3), "interval_s"])  # leader: free air
    assert laps.loc[("BBB", 3), "interval_s"] == pytest.approx(1.5)
    assert np.isnan(laps.loc[("BBB", 1), "interval_start_s"])  # nothing known before lap 1
    assert laps.loc[("BBB", 2), "interval_start_s"] == pytest.approx(1.5)


def test_quali_gap_pct():
    laps = make_laps()
    quali = pd.DataFrame({"Driver": ["AAA", "BBB"], "QualiBestS": [90.0, 90.9]})
    gap = add_quali_prior(laps, quali).groupby("driver")["quali_gap_pct"].first()
    assert gap["AAA"] == pytest.approx(0.0)
    assert gap["BBB"] == pytest.approx(1.0)


def test_circuit_key_aliases_event_before_location():
    aliases = {"Sakhir Grand Prix": "Sakhir Outer"}
    assert circuit_key({"event_name": "Sakhir Grand Prix", "location": "Sakhir"}, aliases) == (
        "Sakhir Outer"
    )
    assert circuit_key({"event_name": "Bahrain Grand Prix", "location": "Sakhir"}, aliases) == (
        "Sakhir"
    )
