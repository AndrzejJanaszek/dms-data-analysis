import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dms_analysis.export import EVENT, PREPARED_KIND, RAW_ACTIVE, RAW_KIND
from dms_analysis.raw import SessionData, to_datetime
from dms_analysis.signals import ACTIVE_LEVEL

INACTIVE_LEVEL = 1 - ACTIVE_LEVEL


@dataclass(frozen=True)
class PreparedSession:
    data: SessionData
    cleaned: pd.DataFrame
    events: pd.DataFrame


def load_sources(db_path: Path) -> pd.DataFrame:
    return _query(db_path, "SELECT DISTINCT source, source_name FROM sessions ORDER BY source_name")


def load_sessions(db_path: Path, source: str) -> pd.DataFrame:
    sessions = _query(
        db_path,
        "SELECT session_id, started_at, first_at, last_at FROM sessions"
        " WHERE source = ? AND first_at IS NOT NULL ORDER BY session_id",
        source,
    )
    for column in ("started_at", "first_at", "last_at"):
        sessions[column] = to_datetime(sessions[column])
    return sessions


def load_variants(db_path: Path) -> pd.DataFrame:
    return _query(db_path, "SELECT variant, description, current FROM variants ORDER BY current DESC, variant")


def load_session(db_path: Path, source: str, session_id: int, variant: str) -> PreparedSession:
    keys = (source, session_id)
    weight = _query(
        db_path,
        "SELECT timestamp, value FROM weight WHERE source = ? AND session_id = ? ORDER BY timestamp",
        *keys,
    )
    temperatures = _query(
        db_path,
        "SELECT timestamp, sensor_address, value FROM temperatures"
        " WHERE source = ? AND session_id = ? ORDER BY timestamp",
        *keys,
    )
    changes = _query(
        db_path,
        "SELECT timestamp, pin, kind, state FROM signal_changes"
        " WHERE variant = ? AND source = ? AND session_id = ? ORDER BY pin, timestamp",
        variant,
        *keys,
    )
    events = _query(
        db_path,
        "SELECT pin, role, start, end, duration, pulse_count, active_share, rejected FROM signal_events"
        " WHERE variant = ? AND source = ? AND session_id = ? ORDER BY pin, start",
        variant,
        *keys,
    )

    for frame in (weight, temperatures, changes):
        frame["timestamp"] = to_datetime(frame["timestamp"])

    raw = changes[changes["kind"] == RAW_KIND]
    prepared = changes[changes["kind"] == PREPARED_KIND]

    return PreparedSession(
        data=SessionData(weight=weight, pins=_levels(raw, RAW_ACTIVE), temperatures=temperatures),
        cleaned=_levels(prepared, EVENT),
        events=_events(events),
    )


def _levels(changes: pd.DataFrame, active_state: int) -> pd.DataFrame:
    value = (changes["state"] == active_state).map({True: ACTIVE_LEVEL, False: INACTIVE_LEVEL})
    return pd.DataFrame({"timestamp": changes["timestamp"], "pin": changes["pin"], "value": value}).reset_index(
        drop=True
    )


def _events(events: pd.DataFrame) -> pd.DataFrame:
    return events.assign(
        start=to_datetime(events["start"]),
        end=to_datetime(events["end"]),
        duration=pd.to_timedelta(events["duration"], unit="s"),
        rejected=events["rejected"].astype(bool),
    )


def _query(db_path: Path, sql: str, *params: object) -> pd.DataFrame:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        return pd.read_sql_query(sql, connection, params=params)
