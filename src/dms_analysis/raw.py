import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TIMEZONE = "Europe/Warsaw"


@dataclass(frozen=True)
class SessionData:
    weight: pd.DataFrame
    pins: pd.DataFrame
    temperatures: pd.DataFrame

    @property
    def end(self) -> pd.Timestamp:
        frames = (self.weight, self.pins, self.temperatures)
        return max(frame["timestamp"].max() for frame in frames if not frame.empty)


def find_databases(data_dir: Path) -> list[Path]:
    return sorted(path for path in data_dir.rglob("*.db") if _has_sessions(path))


def load_sessions(db_path: Path) -> pd.DataFrame:
    query = """
        SELECT s.id AS session_id, s.started_at, d.first_at, d.last_at,
               d.weight_count, d.pin_count, d.temperature_count
        FROM sessions s
        LEFT JOIN (
            SELECT session_id, MIN(timestamp) AS first_at, MAX(timestamp) AS last_at,
                   SUM(kind = 'w') AS weight_count, SUM(kind = 'p') AS pin_count,
                   SUM(kind = 't') AS temperature_count
            FROM (
                SELECT session_id, timestamp, 'w' AS kind FROM measurements
                UNION ALL SELECT session_id, timestamp, 'p' FROM pin_states
                UNION ALL SELECT session_id, timestamp, 't' FROM temperature_measurements
            )
            GROUP BY session_id
        ) d ON d.session_id = s.id
        ORDER BY s.id
    """
    sessions = _query(db_path, query)

    for column in ("started_at", "first_at", "last_at"):
        sessions[column] = _to_datetime(sessions[column])

    sessions["duration"] = sessions["last_at"] - sessions["first_at"]
    return sessions


def load_session(db_path: Path, session_id: int) -> SessionData:
    weight = _query(
        db_path,
        "SELECT timestamp, value FROM measurements WHERE session_id = ? ORDER BY timestamp",
        session_id,
    )
    pins = _query(
        db_path,
        "SELECT timestamp, pin, value FROM pin_states WHERE session_id = ? ORDER BY timestamp",
        session_id,
    )
    temperatures = _query(
        db_path,
        "SELECT timestamp, sensor_address, value FROM temperature_measurements"
        " WHERE session_id = ? ORDER BY timestamp",
        session_id,
    )

    for frame in (weight, pins, temperatures):
        frame["timestamp"] = _to_datetime(frame["timestamp"])

    return SessionData(weight=weight, pins=pins, temperatures=temperatures)


def _has_sessions(db_path: Path) -> bool:
    tables = _query(db_path, "SELECT name FROM sqlite_master WHERE type = 'table'")
    return "sessions" in set(tables["name"])


def _query(db_path: Path, sql: str, *params: object) -> pd.DataFrame:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        return pd.read_sql_query(sql, connection, params=params)


def _to_datetime(seconds: pd.Series) -> pd.Series:
    return pd.to_datetime(seconds, unit="s", utc=True).dt.tz_convert(TIMEZONE)
