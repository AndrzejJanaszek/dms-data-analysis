from pathlib import Path

import pandas as pd

from dms_analysis.data.catalog import Catalog, CatalogError
from dms_analysis.data.dumps import connect_read_only, dump_path
from dms_analysis.raw import TIMEZONE
from dms_analysis.sensors import ERROR_CODES

OLD_SEGMENT_GAP_S = 3600

SESSIONS_SQL = """
    WITH steps AS (
        SELECT session_id, timestamp - LAG(timestamp) OVER (PARTITION BY session_id ORDER BY timestamp) AS step
        FROM measurements
    ),
    pins AS (
        SELECT session_id, value != LAG(value) OVER (PARTITION BY session_id, pin ORDER BY timestamp) AS changed
        FROM pin_states
    )
    SELECT s.id AS session,
           d.first_at, d.last_at,
           COALESCE(d.weight_count, 0) AS weight_count,
           COALESCE(w.min_value, 0) AS weight_min, COALESCE(w.max_value, 0) AS weight_max,
           COALESCE(g.longest_gap, 0) AS longest_gap_s,
           COALESCE(p.pin_changes, 0) AS pin_changes,
           COALESCE(t.temperature_count, 0) AS temperature_count,
           COALESCE(t.valid_temperatures, 0) AS valid_temperatures
    FROM sessions s
    JOIN (
        SELECT session_id, MIN(timestamp) AS first_at, MAX(timestamp) AS last_at, SUM(kind = 'w') AS weight_count
        FROM (
            SELECT session_id, timestamp, 'w' AS kind FROM measurements
            UNION ALL SELECT session_id, timestamp, 'p' FROM pin_states
            UNION ALL SELECT session_id, timestamp, 't' FROM temperature_measurements
        )
        GROUP BY session_id
    ) d ON d.session_id = s.id
    LEFT JOIN (
        SELECT session_id, MIN(value) AS min_value, MAX(value) AS max_value FROM measurements GROUP BY session_id
    ) w ON w.session_id = s.id
    LEFT JOIN (SELECT session_id, MAX(step) AS longest_gap FROM steps GROUP BY session_id) g ON g.session_id = s.id
    LEFT JOIN (SELECT session_id, SUM(changed) AS pin_changes FROM pins GROUP BY session_id) p ON p.session_id = s.id
    LEFT JOIN (
        SELECT session_id, COUNT(*) AS temperature_count,
               SUM(length(sensor_address) = 16 AND value NOT IN ({errors})) AS valid_temperatures
        FROM temperature_measurements GROUP BY session_id
    ) t ON t.session_id = s.id
    ORDER BY s.id
""".format(errors=", ".join(str(code) for code in ERROR_CODES))


def has_sessions(path: Path) -> bool:
    tables = _query(path, "SELECT name FROM sqlite_master WHERE type = 'table'")
    return "sessions" in set(tables["name"])


def dump_sessions(path: Path) -> pd.DataFrame:
    if has_sessions(path):
        return _query(path, SESSIONS_SQL)
    return old_segments(path)


def old_segments(path: Path) -> pd.DataFrame:
    readings = _query(path, "SELECT id, timestamp, value FROM measurements ORDER BY id")
    step = readings["timestamp"].diff()
    segment = ((step < 0) | (step > OLD_SEGMENT_GAP_S)).cumsum() + 1
    readings["step"] = step.where(segment == segment.shift())
    grouped = readings.groupby(segment)

    return pd.DataFrame(
        {
            "session": range(1, grouped.ngroups + 1),
            "first_at": grouped["timestamp"].min().to_numpy(),
            "last_at": grouped["timestamp"].max().to_numpy(),
            "weight_count": grouped.size().to_numpy(),
            "weight_min": grouped["value"].min().to_numpy(),
            "weight_max": grouped["value"].max().to_numpy(),
            "longest_gap_s": grouped["step"].max().fillna(0).to_numpy(),
            "pin_changes": 0,
            "temperature_count": 0,
            "valid_temperatures": 0,
            "first_id": grouped["id"].min().to_numpy(),
            "last_id": grouped["id"].max().to_numpy(),
        }
    )


def uncatalogued(catalog: Catalog, dump_id: str, sessions: pd.DataFrame) -> pd.DataFrame:
    root = catalog.root_dump(dump_id)
    sources = [period.source for period in catalog.periods.values() if catalog.root_dump(period.source.dump) == root]

    def covered(row: pd.Series) -> bool:
        if "first_id" in row:
            return any(
                source.first_id is not None and source.first_id <= row["last_id"] and row["first_id"] <= source.last_id
                for source in sources
            )
        return any(source.session == row["session"] for source in sources)

    return sessions[~sessions.apply(covered, axis=1)] if not sessions.empty else sessions


def sessions_table(sessions: pd.DataFrame) -> pd.DataFrame:
    first = _local(sessions["first_at"])
    last = _local(sessions["last_at"])
    return pd.DataFrame(
        {
            "sesja": sessions["session"],
            "od (zegar)": first.dt.strftime("%Y-%m-%d %H:%M"),
            "do (zegar)": last.dt.strftime("%Y-%m-%d %H:%M"),
            "godziny": ((sessions["last_at"] - sessions["first_at"]) / 3600).round(1),
            "wskazania": sessions["weight_count"],
            "wskazania od–do [kg]": sessions["weight_min"].map("{:.0f}".format)
            + "–"
            + sessions["weight_max"].map("{:.0f}".format),
            "zmiany pinów": sessions["pin_changes"],
            "temperatury": sessions["temperature_count"],
            "najdłuższa przerwa [s]": sessions["longest_gap_s"].round(1),
        }
    )


def period_draft(catalog: Catalog, dump_id: str, sessions: pd.DataFrame, site_id: str, session: int) -> str:
    matching = sessions[sessions["session"] == session]
    if matching.empty:
        raise CatalogError(f"zrzut {dump_id} nie ma sesji (odcinka) {session}")
    row = matching.iloc[0]

    site_periods = [period for period in catalog.periods.values() if period.site == site_id]
    number = max((period.number for period in site_periods), default=0) + 1
    hardware = catalog.sites[site_id].hardware if site_id in catalog.sites else {}
    hardware_id = list(hardware)[-1] if hardware else "h1"
    start, end = _local(pd.Series([row["first_at"], row["last_at"]]))

    if "first_id" in row:
        source = (
            f'source = {{ dump = "{dump_id}", first_id = {int(row["first_id"])}, last_id = {int(row["last_id"])} }}'
        )
    else:
        source = f'source = {{ dump = "{dump_id}", session = {int(row["session"])} }}'

    return "\n".join(
        [
            "# szkic: czas wg zegara urządzenia; ustaw time, clock_offset_h, time_uncertainty_h,",
            "# przelicz start i end (rzeczywisty = zegar + przesunięcie), datę w id i stany kanałów",
            "[[period]]",
            f'id = "{number:04d}-{start:%Y-%m-%d}"',
            'legacy_name = ""',
            f'hardware = "{hardware_id}"',
            source,
            'time = "real"',
            "clock_offset_h = 0",
            "time_uncertainty_h = 0",
            f"start = {toml_datetime(start)}",
            f"end = {toml_datetime(end)}",
            "checked = false",
            'note = ""',
            "",
            "[period.channels]",
            'weight = "ok"',
            f'dispense = "{_pins_guess(row)}"',
            f'refill = "{_pins_guess(row)}"',
            f'temperatures = "{_temperatures_guess(row)}"',
        ]
    )


def toml_datetime(moment: pd.Timestamp) -> str:
    return moment.tz_convert(TIMEZONE).floor("s").isoformat()


def _pins_guess(row: pd.Series) -> str:
    return "ok" if row["pin_changes"] > 0 else "not_connected"


def _temperatures_guess(row: pd.Series) -> str:
    if row["valid_temperatures"] > 0:
        return "ok"
    return "no_values" if row["temperature_count"] > 0 else "not_connected"


def _local(seconds: pd.Series) -> pd.Series:
    return pd.to_datetime(seconds, unit="s", utc=True).dt.tz_convert(TIMEZONE)


def _query(path: Path, sql: str) -> pd.DataFrame:
    connection = connect_read_only(path)
    try:
        return pd.read_sql_query(sql, connection)
    finally:
        connection.close()


def sessions_for_dump(catalog: Catalog, dump_id: str) -> pd.DataFrame:
    if dump_id not in catalog.dumps:
        raise CatalogError(f"nie ma zrzutu {dump_id} w raw.toml")
    path = dump_path(catalog.dumps[dump_id])
    if not path.exists():
        raise CatalogError(f"brak pliku zrzutu {path} (pobierz go z Drive)")
    return dump_sessions(path)
