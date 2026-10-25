"""OpenF1 adapter (https://openf1.org), used for 2023 onwards.

OpenF1 serves the same F1 live-timing feed as FastF1 but in a different shape. This module
rebuilds FastF1-style laps from it:

- lap end ``Time`` = ``date_start + lap_duration`` (falls back to the next lap's start)
- tyre ``Stint``/``Compound``/``TyreLife``/``FreshTyre``: boundaries from pit out-laps, compound
  and starting age from OpenF1 stint records (whose lap ranges are inconsistent between races)
- ``PitInTime``/``PitOutTime`` from the pit endpoint and ``is_pit_out_lap``
- ``TrackStatus`` (FastF1 codes) replayed from race-control messages
- ``Position`` = order of crossing the line on each lap; weather joined at lap start

Classification and qualifying come from Jolpica. Responses are cached on disk and requests
are throttled to the free tier (30 per minute).
"""

import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from strategist.data.sources import LAP_COLUMNS, RaceData, TimingUnavailableError, jolpica

log = logging.getLogger(__name__)

BASE_URL = "https://api.openf1.org/v1"
RACE_ENDPOINTS = ("laps", "stints", "pit", "race_control", "weather", "drivers")

# race-control message (prefix) or track flag -> FastF1 track status code
SAFETY_CAR_MESSAGES = {
    "VIRTUAL SAFETY CAR DEPLOYED": "6",
    "VIRTUAL SAFETY CAR ENDING": "7",
    "SAFETY CAR DEPLOYED": "4",
}
TRACK_FLAGS = {"RED": "5", "CLEAR": "1", "GREEN": "1"}


class OpenF1Client:
    def __init__(self, cache_dir: str | Path, min_interval_s: float = 2.1, retries: int = 4):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval_s = min_interval_s
        self.retries = retries
        self._last_request = 0.0
        self._http = requests.Session()

    def get(self, endpoint: str, **params) -> list[dict]:
        key = "_".join([endpoint, *(f"{k}-{v}" for k, v in sorted(params.items()))])
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

        for attempt in range(self.retries + 1):
            wait = self._last_request + self.min_interval_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            resp = self._http.get(f"{BASE_URL}/{endpoint}", params=params, timeout=60)
            if resp.status_code == 404:  # OpenF1's "no results"
                data = []
                break
            if resp.status_code == 429 or resp.status_code >= 500:
                backoff = 15 * 2**attempt
                log.warning("OpenF1 %s %s -> %d, retrying in %ds",
                            endpoint, params, resp.status_code, backoff)
                time.sleep(backoff)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        else:
            raise RuntimeError(f"OpenF1 {endpoint} {params}: gave up after {self.retries} retries")

        path.write_text(json.dumps(data), encoding="utf-8")
        return data


class OpenF1Source:
    name = "openf1"

    def __init__(self, cache_dir: str | Path, jolpica_cache: str | Path):
        self.client = OpenF1Client(cache_dir)
        self.jolpica_cache = jolpica_cache
        self._sessions: dict[int, pd.DataFrame] = {}

    def fetch(self, year: int, event: pd.Series) -> RaceData:
        rnd = int(event["RoundNumber"])
        session = self._race_session(year, event)
        key = int(session["session_key"])
        raw = {ep: pd.DataFrame(self.client.get(ep, session_key=key)) for ep in RACE_ENDPOINTS}
        if raw["laps"].empty:
            raise TimingUnavailableError(f"OpenF1 has no laps for session {key}")

        origin = pd.Timestamp(session["date_start"])
        laps = build_laps(raw, origin)
        results = jolpica.race_results(year, rnd, self.jolpica_cache)
        team_by_number = laps.drop_duplicates("DriverNumber").set_index("DriverNumber")["Team"]
        results["TeamName"] = results["DriverNumber"].map(team_by_number).fillna(results["TeamName"])

        quali, quali_source = jolpica.quali_best(year, rnd, self.jolpica_cache), "jolpica"
        if len(quali) < len(results) // 2:
            log.warning("%s R%02d: Jolpica has no qualifying times; using OpenF1 laps", year, rnd)
            quali, quali_source = self._quali_best(int(session["meeting_key"])), "openf1"

        meta = {
            # winner's lap count; the originally scheduled distance isn't in OpenF1
            "total_laps": int(results["Laps"].max()),
            "laps_completed": int(laps["LapNumber"].max()),
            "quali_source": quali_source,
            "openf1_session_key": key,
            "openf1_meeting_key": int(session["meeting_key"]),
        }
        return RaceData(laps=laps, results=results, quali=quali, meta=meta)

    def _quali_best(self, meeting_key: int) -> pd.DataFrame:
        sessions = self.client.get("sessions", meeting_key=meeting_key, session_name="Qualifying")
        if not sessions:
            return pd.DataFrame({"Driver": pd.Series(dtype=str), "QualiBestS": pd.Series(dtype=float)})
        key = sessions[0]["session_key"]
        return quali_best_laps(
            pd.DataFrame(self.client.get("laps", session_key=key)),
            pd.DataFrame(self.client.get("race_control", session_key=key)),
            pd.DataFrame(self.client.get("drivers", session_key=key)),
        )

    def _race_session(self, year: int, event: pd.Series) -> pd.Series:
        if year not in self._sessions:
            sessions = pd.DataFrame(self.client.get("sessions", year=year, session_name="Race"))
            sessions["date_start"] = pd.to_datetime(sessions["date_start"], utc=True)
            self._sessions[year] = sessions
        sessions = self._sessions[year]
        race_start = pd.Timestamp(event["Session5DateUtc"]).tz_localize("UTC")
        delta = (sessions["date_start"] - race_start).abs()
        if delta.empty or delta.min() > pd.Timedelta(days=1):
            raise TimingUnavailableError(f"no OpenF1 race session near {race_start:%Y-%m-%d}")
        return sessions.loc[delta.idxmin()]


DELETED_LAP = re.compile(r"CAR (\d+) \(\w+\) TIME (\d+):(\d+\.\d+) DELETED")


def quali_best_laps(laps: pd.DataFrame, race_control: pd.DataFrame,
                    drivers: pd.DataFrame) -> pd.DataFrame:
    """Best qualifying lap per driver, skipping laps race control deleted (matched by time)."""
    deleted = set()
    for msg in race_control.get("message", pd.Series(dtype=str)).dropna():
        if m := DELETED_LAP.search(msg):
            deleted.add((int(m[1]), round(int(m[2]) * 60 + float(m[3]), 3)))
    timed = laps.dropna(subset=["lap_duration"])
    keep = [(int(d), round(t, 3)) not in deleted
            for d, t in zip(timed["driver_number"], timed["lap_duration"], strict=True)]
    best = timed[keep].groupby("driver_number")["lap_duration"].min()
    acronym = drivers.drop_duplicates("driver_number", keep="last").set_index("driver_number")
    return pd.DataFrame({"Driver": best.index.map(acronym["name_acronym"]),
                         "QualiBestS": best.to_numpy()}).dropna()


def seconds_since(dates: pd.Series, origin: pd.Timestamp) -> pd.Series:
    return (pd.to_datetime(dates, utc=True, format="ISO8601") - origin).dt.total_seconds()


def track_status_events(race_control: pd.DataFrame, origin: pd.Timestamp) -> list[tuple[float, str]]:
    """Ordered (seconds, code) track-state changes replayed from race-control messages."""
    rc = race_control.reindex(columns=["date", "category", "flag", "scope", "message"])
    rc = rc.assign(t=seconds_since(rc["date"], origin)).sort_values("t")
    events = []
    for m in rc.itertuples():
        code = None
        if m.category == "SafetyCar":
            msg = str(m.message).upper()
            code = next((c for prefix, c in SAFETY_CAR_MESSAGES.items()
                         if msg.startswith(prefix)), None)
        elif m.category == "Flag" and m.scope == "Track":
            code = TRACK_FLAGS.get(m.flag)
        if code is not None:
            events.append((m.t, code))
    return events


def lap_track_status(starts, ends, events: list[tuple[float, str]]) -> list[str]:
    """FastF1-style status string per lap: state at lap start, then each change during it."""
    times = np.array([t for t, _ in events], dtype=float)
    codes = [c for _, c in events]
    out = []
    for start, end in zip(starts, ends, strict=True):
        start = end if np.isnan(start) else start
        end = start if np.isnan(end) else end
        if np.isnan(start):
            out.append("")
            continue
        i0 = np.searchsorted(times, start, side="right")
        i1 = np.searchsorted(times, end, side="right")
        seq = [codes[i0 - 1] if i0 > 0 else "1", *codes[i0:i1]]
        out.append("".join(dict.fromkeys(seq)))
    return out


TYRE_COLUMNS = ["DriverNumber", "LapNumber", "Stint", "Compound", "TyreLife", "FreshTyre"]


def match_stint_records(records: pd.DataFrame, starts: list[int]) -> list[int] | None:
    """Index of the OpenF1 stint record supplying each pit-defined stint, or None if they conflict.

    OpenF1 stint lap ranges are unreliable (some races start a stint on the in-lap, some on
    the out-lap; records are occasionally split), so records only provide compound and
    starting age. Equal counts pair in order; otherwise each stint takes the record starting
    nearest to it, and a left-over record is only tolerated if it is within a lap of a pit
    stop or keeps the same compound (a split record).
    """
    if records.empty:
        return None
    if len(records) == len(starts):
        return list(range(len(records)))
    rec_starts = records["lap_start"].to_numpy(float)
    chosen = [int(np.argmin(np.abs(rec_starts - s))) for s in starts]
    for i, rec in enumerate(records.itertuples()):
        if i in chosen:
            continue
        k = max(int(np.searchsorted(starts, rec.lap_start, side="right")) - 1, 0)
        near_stop = min(abs(rec.lap_start - s) for s in starts) <= 1
        if not near_stop and rec.compound != records.iloc[chosen[k]].compound:
            return None
    return chosen


def tyre_laps(stints: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    """Per-lap stint, compound and tyre age, with stint boundaries anchored on pit out-laps.
    FastF1 TyreLife convention: a fresh tyre's first lap is 1. Drivers whose stint records
    contradict their pit stops get no tyre data (and are excluded downstream)."""
    rows = []
    for drv, g in laps.groupby("DriverNumber"):
        last_lap = int(g["LapNumber"].max())
        outs = g.loc[g["PitOutTime"].notna() & g["LapNumber"].gt(1), "LapNumber"].astype(int)
        starts = [1, *sorted(set(outs))]
        records = (stints[stints["driver_number"].astype(str) == drv]
                   .dropna(subset=["lap_start"]).sort_values("stint_number"))
        chosen = match_stint_records(records, starts)
        if chosen is None:
            log.warning("driver %s: stint records contradict pit stops; no tyre data", drv)
            continue
        ends = [*starts[1:], last_lap + 1]
        stint_no, age0, first = 0, np.nan, 1
        for k, (start, end) in enumerate(zip(starts, ends, strict=True)):
            rec = records.iloc[chosen[k]]
            if k > 0 and chosen[k] == chosen[k - 1]:
                pass  # same tyres after the stop (drive-through/stop-go): age carries on
            else:
                stint_no, age0, first = stint_no + 1, rec.tyre_age_at_start, start
            for lap in range(start, end):
                rows.append({
                    "DriverNumber": drv,
                    "LapNumber": float(lap),
                    "Stint": float(stint_no),
                    "Compound": rec.compound,
                    "TyreLife": age0 + lap - first + 1 if pd.notna(age0) else np.nan,
                    "FreshTyre": bool(age0 == 0) if pd.notna(age0) else None,
                })
    return pd.DataFrame(rows, columns=TYRE_COLUMNS)


def build_laps(raw: dict[str, pd.DataFrame], origin: pd.Timestamp) -> pd.DataFrame:
    src = raw["laps"].sort_values(["driver_number", "lap_number"]).reset_index(drop=True)
    start = seconds_since(src["date_start"], origin)
    end = (start + src["lap_duration"]).fillna(start.groupby(src["driver_number"]).shift(-1))

    laps = pd.DataFrame({
        "DriverNumber": src["driver_number"].astype(str),
        "LapNumber": src["lap_number"].astype(float),
        "LapTime": src["lap_duration"],
        "Time": end,
        "LapStartTime": start,
        "Sector1Time": src["duration_sector_1"],
        "Sector2Time": src["duration_sector_2"],
        "Sector3Time": src["duration_sector_3"],
        "PitOutTime": start.where(src["is_pit_out_lap"].eq(True)),
        "IsAccurate": src["lap_duration"].notna(),
        "FastF1Generated": False,
    })

    # in-lap: the pit endpoint's lap, or the lap before an out-lap
    pit = raw["pit"]
    in_laps = set()
    if not pit.empty:
        in_laps |= set(zip(pit["driver_number"].astype(str), pit["lap_number"].astype(float)))
    out_laps = laps.loc[laps["PitOutTime"].notna() & laps["LapNumber"].gt(1)]
    in_laps |= set(zip(out_laps["DriverNumber"], out_laps["LapNumber"] - 1))
    is_in = [k in in_laps for k in zip(laps["DriverNumber"], laps["LapNumber"])]
    laps["PitInTime"] = laps["Time"].where(is_in)

    drivers = raw["drivers"].drop_duplicates("driver_number", keep="last")
    drivers = drivers.assign(DriverNumber=drivers["driver_number"].astype(str)).set_index("DriverNumber")
    laps["Driver"] = laps["DriverNumber"].map(drivers["name_acronym"])
    laps["Team"] = laps["DriverNumber"].map(drivers["team_name"])

    laps = laps.merge(tyre_laps(raw["stints"], laps), on=["DriverNumber", "LapNumber"], how="left")
    events = track_status_events(raw["race_control"], origin)
    laps["TrackStatus"] = lap_track_status(laps["LapStartTime"], laps["Time"], events)
    laps["Position"] = line_positions(laps)
    laps = _join_weather(laps, raw["weather"], origin)
    return laps.reindex(columns=LAP_COLUMNS)


def line_positions(laps: pd.DataFrame) -> pd.Series:
    """Race position at the end of each lap: order of crossing the line on that lap.

    Derived from timing rather than OpenF1's position feed, which lags the line crossing.
    Everyone ahead of a car has completed the same lap before it, so lapped cars rank correctly.
    """
    return laps.groupby("LapNumber")["Time"].rank(method="first")


def _join_weather(laps: pd.DataFrame, weather: pd.DataFrame, origin) -> pd.DataFrame:
    """Weather sample nearest the start of each lap (samples are once a minute)."""
    if weather.empty:
        return laps
    w = pd.DataFrame({
        "t": seconds_since(weather["date"], origin),
        "AirTemp": weather["air_temperature"],
        "TrackTemp": weather["track_temperature"],
        "Humidity": weather["humidity"],
        "Pressure": weather["pressure"],
        "Rainfall": weather["rainfall"].astype(float) > 0,
        "WindSpeed": weather["wind_speed"],
    }).sort_values("t")
    key = laps["LapStartTime"].fillna(laps["Time"])
    known = key.notna()
    left = laps.loc[known, []].assign(t=key[known]).sort_values("t")
    joined = pd.merge_asof(left, w, on="t", direction="nearest")
    joined.index = left.index
    for col in w.columns.drop("t"):
        laps[col] = joined[col]
    return laps
