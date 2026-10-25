"""FastF1 adapter.

Without an F1 TV sign-in FastF1 falls back to its community mirror, which covers 2021–2022.
"""

import logging

import fastf1
import pandas as pd
from fastf1.exceptions import DataNotLoadedError, RateLimitExceededError

from strategist.data.sources import (
    RESULT_COLUMNS,
    WEATHER_COLUMNS,
    RaceData,
    TimingUnavailableError,
    timedeltas_to_seconds,
)

log = logging.getLogger(__name__)


class FastF1Source:
    name = "fastf1"

    def fetch(self, year: int, event: pd.Series) -> RaceData:
        rnd = int(event["RoundNumber"])
        laps, results, meta = self._race(year, rnd)
        try:
            quali = self._quali_best(year, rnd)
        except RateLimitExceededError:
            raise
        except Exception as exc:
            log.warning("%s R%02d: qualifying unavailable (%s)", year, rnd, exc)
            quali = pd.DataFrame({"Driver": pd.Series(dtype=str),
                                  "QualiBestS": pd.Series(dtype=float)})
        return RaceData(laps=laps, results=results, quali=quali, meta=meta)

    def _race(self, year: int, rnd: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        session = fastf1.get_session(year, rnd, "R")
        session.load(laps=True, telemetry=False, weather=True, messages=False)

        try:
            laps = pd.DataFrame(session.laps).reset_index(drop=True)
        except DataNotLoadedError:
            raise TimingUnavailableError(
                "no timing data: livetiming.formula1.com needs an F1 TV sign-in "
                "(F1_USERNAME/F1_PASSWORD) and the FastF1 mirror lacks this session"
            ) from None
        try:
            weather = session.laps.get_weather_data().reset_index(drop=True)
            laps = pd.concat([laps, weather.reindex(columns=WEATHER_COLUMNS)], axis=1)
        except Exception as exc:  # weather feed is missing for a few sessions
            log.warning("%s R%02d: no weather data (%s)", year, rnd, exc)
            for col in WEATHER_COLUMNS:
                laps[col] = pd.NA
        laps = timedeltas_to_seconds(laps)

        results = timedeltas_to_seconds(
            pd.DataFrame(session.results).reindex(columns=RESULT_COLUMNS).reset_index(drop=True)
        )
        meta = {
            "total_laps": session.total_laps,
            "laps_completed": int(laps["LapNumber"].max()) if len(laps) else 0,
        }
        return laps, results, meta

    @staticmethod
    def _quali_best(year: int, rnd: int) -> pd.DataFrame:
        session = fastf1.get_session(year, rnd, "Q")
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps
        valid = laps[laps["Deleted"].ne(True) & laps["LapTime"].notna()]
        best = valid.groupby("Driver")["LapTime"].min().dt.total_seconds()
        return best.rename("QualiBestS").reset_index()
