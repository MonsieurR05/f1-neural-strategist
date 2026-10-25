"""FastF1 adapter.

Without an F1 TV sign-in FastF1 falls back to its community mirror, which covers 2021–2022.
"""

import logging

import fastf1
import pandas as pd
from fastf1.exceptions import DataNotLoadedError

from strategist.data.sources import (
    RESULT_COLUMNS,
    WEATHER_COLUMNS,
    RaceData,
    TimingUnavailableError,
    jolpica,
    timedeltas_to_seconds,
)

log = logging.getLogger(__name__)


class FastF1Source:
    name = "fastf1"

    def fetch(self, year: int, event: pd.Series) -> RaceData:
        rnd = int(event["RoundNumber"])
        laps, results, meta = self._race(year, rnd)
        # Jolpica's Q1-Q3 times: one request instead of loading the whole qualifying
        # session, and the same definition the OpenF1 adapter uses.
        return RaceData(laps=laps, results=results, quali=jolpica.quali_best(year, rnd), meta=meta)

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
