import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from dms_analysis.data.catalog import CANONICAL_PINS, ROLE_NAMES, SIGNAL_CHANNELS, Catalog, Hardware, Period
from dms_analysis.data.extract import unpacked
from dms_analysis.data.paths import ALL_PERIODS_DB, PREPARED_DIR, PROJECT_DIR, prepared_path
from dms_analysis.export import events_rows, signal_changes
from dms_analysis.raw import TIMEZONE
from dms_analysis.sensors import ADDRESS_LENGTH, ERROR_CODES
from dms_analysis.signals import CURRENT_VARIANT, VARIANTS, SignalEvents, Variant, detect_events

GAP_SECONDS = 60
START_TOLERANCE_S = 1.0
DETECTED_STATES = ("ok", "flickering")

TRUNCATED_ADDRESS = "ucięty adres"
ERROR_CODE = "kod błędu"
UNKNOWN_SENSOR = "czujnik spoza sprzętu"

TIME_INDEXES = {
    "weight": "timestamp",
    "temperatures": "timestamp",
    "removed": "timestamp",
    "signal_changes": "pin, timestamp",
    "signal_events": "pin, start",
}
RAW_DTYPES = {"timestamp": "float64", "value": "float64", "pin": "int64", "sensor_address": "str"}
TABLES = ("period", "weight", "temperatures", "removed", "signal_changes", "signal_events", "issues", "parts")
EMPTY_COLUMNS = {
    "signal_changes": {"pin": "int64", "kind": "str", "timestamp": "float64", "state": "int64"},
    "signal_events": {
        "pin": "int64",
        "role": "str",
        "start": "float64",
        "end": "float64",
        "duration": "float64",
        "pulse_count": "int64",
        "active_share": "float64",
        "rejected": "int64",
        "synthetic": "int64",
    },
    "issues": {"channel": "str", "kind": "str", "start": "float64", "end": "float64", "note": "str"},
    "parts": {"part": "int64", "start": "float64", "end": "float64"},
}


@dataclass(frozen=True)
class RawPeriod:
    weight: pd.DataFrame
    pins: pd.DataFrame
    temperatures: pd.DataFrame

    @property
    def end(self) -> pd.Timestamp:
        return max(frame["timestamp"].max() for frame in self.frames if not frame.empty)

    @property
    def start(self) -> pd.Timestamp:
        return min(frame["timestamp"].min() for frame in self.frames if not frame.empty)

    @property
    def frames(self) -> tuple[pd.DataFrame, ...]:
        return (self.weight, self.pins, self.temperatures)


def prepare(catalog: Catalog, period: Period) -> list[str]:
    target = prepared_path(period.full_id)
    target.parent.mkdir(parents=True, exist_ok=True)

    with unpacked(period.full_id, target.parent) as database:
        raw = load_raw(database, period.clock_offset_h)

    tables = prepared_tables(period, raw)
    write_database(target, tables)
    return catalog_warnings(period, raw)


def load_raw(database: Path, clock_offset_h: float) -> RawPeriod:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        names = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        frames = [
            _read(connection, names, "measurements", "timestamp, value"),
            _read(connection, names, "pin_states", "timestamp, pin, value"),
            _read(connection, names, "temperature_measurements", "timestamp, sensor_address, value"),
        ]
    finally:
        connection.close()

    shift = clock_offset_h * 3600
    return RawPeriod(*(frame.assign(timestamp=to_local(frame["timestamp"] + shift)) for frame in frames))


def _read(connection: sqlite3.Connection, names: set[str], table: str, columns: str) -> pd.DataFrame:
    if table not in names:
        return pd.DataFrame({name: pd.Series(dtype=RAW_DTYPES[name]) for name in columns.split(", ")})
    return pd.read_sql_query(f"SELECT {columns} FROM {table} ORDER BY timestamp, id", connection)


def to_local(seconds: pd.Series) -> pd.Series:
    return pd.to_datetime(seconds, unit="s", utc=True).dt.tz_convert(TIMEZONE)


def epoch_seconds(moments: pd.Series) -> pd.Series:
    return (moments - pd.Timestamp(0, tz="UTC")).dt.total_seconds()


def prepared_tables(period: Period, raw: RawPeriod) -> dict[str, pd.DataFrame]:
    temperatures, removed = split_temperatures(raw.temperatures, period.hardware)
    signal_events = period_signal_events(period, raw)
    recorded = recorded_pins(period, raw.pins)

    return {
        "period": period_row(period),
        "weight": raw.weight.assign(timestamp=epoch_seconds(raw.weight["timestamp"])),
        "temperatures": temperatures,
        "removed": removed,
        "signal_changes": _changes(recorded, signal_events),
        "signal_events": _events(signal_events),
        "issues": issues_rows(period),
        "parts": parts(epoch_seconds(raw.weight["timestamp"])),
    }


def split_temperatures(temperatures: pd.DataFrame, hardware: Hardware) -> tuple[pd.DataFrame, pd.DataFrame]:
    numbers = {sensor.address: sensor.number for sensor in hardware.temperature_sensors}
    address = temperatures["sensor_address"].astype(str)
    reason = np.select(
        [
            address.str.len() != ADDRESS_LENGTH,
            temperatures["value"].isin(ERROR_CODES),
            ~address.isin(numbers),
        ],
        [TRUNCATED_ADDRESS, ERROR_CODE, UNKNOWN_SENSOR],
        default="",
    )
    kept = temperatures[reason == ""]
    dropped = temperatures[reason != ""]

    return (
        pd.DataFrame(
            {
                "timestamp": epoch_seconds(kept["timestamp"]),
                "sensor": kept["sensor_address"].map(numbers).astype(int),
                "value": kept["value"],
            }
        ),
        pd.DataFrame(
            {
                "kind": reason[reason != ""],
                "timestamp": epoch_seconds(dropped["timestamp"]),
                "sensor_address": dropped["sensor_address"],
                "value": dropped["value"],
            }
        ),
    )


def recorded_pins(period: Period, pins: pd.DataFrame) -> pd.DataFrame:
    connected = [pin for pin, channel in period.hardware.pins.items() if period.channels[channel] != "not_connected"]
    return pins[pins["pin"].isin(connected)].assign(pin=lambda frame: frame["pin"].astype(int))


def period_signal_events(period: Period, raw: RawPeriod) -> SignalEvents:
    hardware = period.hardware
    recorded = recorded_pins(period, raw.pins)
    present = set(recorded["pin"])
    detected_pins = [
        pin
        for channel in SIGNAL_CHANNELS
        if (pin := hardware.pin_for(channel)) is not None
        and period.channels[channel] in DETECTED_STATES
        and pin in present
    ]

    if detected_pins:
        variant = hardware_variant(VARIANTS[CURRENT_VARIANT], hardware)
        detected = detect_events(recorded, raw.end, pin_numbers=sorted(detected_pins), variant=variant)
        events, too_short, pulses = detected.events, detected.too_short, detected.pulses
    else:
        events = too_short = empty_events()
        pulses = pd.DataFrame(columns=["start", "end", "pin"])

    roles = {pin: ROLE_NAMES[channel] for pin, channel in hardware.pins.items()}
    events = pd.concat([events.assign(synthetic=False), manual_rows(period, raw.end)], ignore_index=True)

    return SignalEvents(
        events=events.assign(role=events["pin"].map(roles)).sort_values(["pin", "start"]).reset_index(drop=True),
        too_short=too_short.assign(role=too_short["pin"].map(roles), synthetic=False),
        pulses=pulses,
    )


def hardware_variant(variant: Variant, hardware: Hardware) -> Variant:
    rules = {pin: variant.rules(CANONICAL_PINS[channel]) for pin, channel in hardware.pins.items()}
    return Variant(variant.description, variant.default, rules)


def manual_rows(period: Period, period_end: pd.Timestamp) -> pd.DataFrame:
    if not period.manual_events:
        return empty_events()

    events = period.manual_events
    rows = pd.DataFrame(
        {
            "role": [ROLE_NAMES[event.channel] for event in events],
            "pin": [period.hardware.pin_for(event.channel) for event in events],
            "start": [_timestamp(event.start) for event in events],
            "end": [period_end if event.end is None else _timestamp(event.end) for event in events],
            "open_end": [event.end is None for event in events],
        }
    )
    duration = rows["end"] - rows["start"]
    return rows.assign(
        duration=duration,
        pulse_count=0,
        active_time=duration,
        active_share=1.0,
        open_start=False,
        synthetic=True,
    )


def empty_events() -> pd.DataFrame:
    moment = f"datetime64[ns, {TIMEZONE}]"
    return pd.DataFrame(
        {
            "role": pd.Series(dtype=str),
            "pin": pd.Series(dtype=int),
            "start": pd.Series(dtype=moment),
            "end": pd.Series(dtype=moment),
            "duration": pd.Series(dtype="timedelta64[ns]"),
            "pulse_count": pd.Series(dtype=int),
            "active_time": pd.Series(dtype="timedelta64[ns]"),
            "active_share": pd.Series(dtype=float),
            "open_start": pd.Series(dtype=bool),
            "open_end": pd.Series(dtype=bool),
            "synthetic": pd.Series(dtype=bool),
        }
    )


def _changes(recorded: pd.DataFrame, signal_events: SignalEvents) -> pd.DataFrame:
    if recorded.empty and signal_events.events.empty:
        return typed_empty("signal_changes")
    return signal_changes(recorded, signal_events)


def _events(signal_events: SignalEvents) -> pd.DataFrame:
    if signal_events.events.empty and signal_events.too_short.empty:
        return typed_empty("signal_events")
    return events_rows(signal_events)


def typed_empty(table: str) -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in EMPTY_COLUMNS[table].items()})


def issues_rows(period: Period) -> pd.DataFrame:
    if not period.issues:
        return typed_empty("issues")
    return pd.DataFrame(
        {
            "channel": [issue.channel for issue in period.issues],
            "kind": [issue.kind for issue in period.issues],
            "start": [issue.start.timestamp() for issue in period.issues],
            "end": [issue.end.timestamp() for issue in period.issues],
            "note": [issue.note for issue in period.issues],
        }
    )


def parts(seconds: pd.Series) -> pd.DataFrame:
    if seconds.empty:
        return typed_empty("parts")
    number = (seconds.diff() > GAP_SECONDS).cumsum() + 1
    grouped = seconds.groupby(number)
    return pd.DataFrame(
        {"part": grouped.min().index, "start": grouped.min().to_numpy(), "end": grouped.max().to_numpy()}
    )


def period_row(period: Period) -> pd.DataFrame:
    source = period.source
    variant = VARIANTS[CURRENT_VARIANT]
    return pd.DataFrame(
        [
            {
                "id": period.full_id,
                "site": period.site,
                "period_id": period.id,
                "label": period_label(period),
                "legacy_name": period.legacy_name,
                "hardware": period.hardware.id,
                "dump": source.dump,
                "session": source.session,
                "first_id": source.first_id,
                "last_id": source.last_id,
                "time": period.time,
                "clock_offset_h": period.clock_offset_h,
                "time_uncertainty_h": period.time_uncertainty_h,
                "start": period.start.timestamp(),
                "end": period.end.timestamp(),
                "checked": int(period.checked),
                "note": period.note,
                **{f"channel_{name}": state for name, state in period.channels.items()},
                "variant": CURRENT_VARIANT,
                "variant_description": variant.description,
                "catalog_sha256": period.catalog_sha256,
                "commit": source_commit(),
                "prepared_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
            }
        ]
    ).astype({"session": "Int64", "first_id": "Int64", "last_id": "Int64"})


def period_label(period: Period) -> str:
    return f"{period.full_id} — {time_label(period)}"


def time_label(period: Period) -> str:
    if period.time == "real":
        return "czas rzeczywisty"
    if period.time == "estimated":
        return f"czas szacowany ±{period.time_uncertainty_h:g} h".replace(".", ",")
    return "czas zegara"


def source_commit() -> str:
    commit = _git("rev-parse", "--short", "HEAD")
    dirty = _git("status", "--porcelain", "--", "src", "catalog") != ""
    return commit + ("+zmiany" if dirty else "")


def _git(*arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=PROJECT_DIR, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def write_database(target: Path, tables: dict[str, pd.DataFrame]) -> None:
    with tempfile.TemporaryDirectory(dir=target.parent) as working:
        database = Path(working) / target.name
        with sqlite3.connect(database) as connection:
            for name, table in tables.items():
                table.to_sql(name, connection, index=False)
            for name, columns in TIME_INDEXES.items():
                if name in tables:
                    connection.execute(f"CREATE INDEX {name}_time ON {name} ({columns})")
        database.replace(target)


def catalog_warnings(period: Period, raw: RawPeriod) -> list[str]:
    warnings = []
    for name, catalog_time, data_time in (("start", period.start, raw.start), ("end", period.end, raw.end)):
        difference = abs(_timestamp(catalog_time) - data_time).total_seconds()
        if difference > START_TOLERANCE_S:
            found = data_time.floor("s").isoformat()
            warnings.append(
                f"{period.full_id}: {name} w katalogu różni się od danych o {difference:.0f} s (w danych {found})"
            )
    return warnings


def stored_catalog_sha256(database: Path) -> str:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        return connection.execute("SELECT catalog_sha256 FROM period").fetchone()[0]
    finally:
        connection.close()


def is_stale(period: Period) -> bool:
    path = prepared_path(period.full_id)
    return not path.exists() or stored_catalog_sha256(path) != period.catalog_sha256


def build_all_periods(catalog: Catalog) -> list[str]:
    included = [period for period in catalog.periods.values() if prepared_path(period.full_id).exists()]
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=PREPARED_DIR) as working:
        database = Path(working) / ALL_PERIODS_DB.name
        connection = sqlite3.connect(database)
        for period in included:
            _append_period(connection, period.full_id)
        for name, columns in TIME_INDEXES.items():
            connection.execute(f"CREATE INDEX {name}_time ON {name} (period, {columns})")
        connection.commit()
        connection.close()
        database.replace(ALL_PERIODS_DB)

    return [period.full_id for period in included]


def _append_period(connection: sqlite3.Connection, full_id: str) -> None:
    connection.execute("ATTACH DATABASE ? AS period_db", (f"file:{prepared_path(full_id)}?mode=ro",))
    existing = {row[0] for row in connection.execute("SELECT name FROM main.sqlite_master WHERE type = 'table'")}
    for table in TABLES:
        if table in existing:
            columns = ", ".join(f'"{row[1]}"' for row in connection.execute(f"PRAGMA period_db.table_info({table})"))
            connection.execute(
                f"INSERT INTO main.{table} (period, {columns}) SELECT ?, {columns} FROM period_db.{table}", (full_id,)
            )
        else:
            connection.execute(f"CREATE TABLE main.{table} AS SELECT ? AS period, * FROM period_db.{table}", (full_id,))
    connection.commit()
    connection.execute("DETACH DATABASE period_db")


def _timestamp(moment: datetime) -> pd.Timestamp:
    return pd.Timestamp(moment).tz_convert(TIMEZONE)
