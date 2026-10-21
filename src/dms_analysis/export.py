import sqlite3
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from dms_analysis.raw import load_session, load_sessions
from dms_analysis.signals import ACTIVE_LEVEL, CURRENT_VARIANT, PIN_ROLES, VARIANTS, SignalEvents, detect_events

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
GRAFANA_DATASOURCES = PROJECT_DIR / "grafana" / "provisioning" / "datasources" / "sqlite.yaml"
GRAFANA_DATA_DIR = "/data"
PREPARED_DB = DATA_DIR / "prepared" / "prepared.db"

INACTIVE = 0
RAW_ACTIVE = 1
EVENT = 2
REJECTED = 3

TABLES = {
    "sessions": "source, session_id",
    "weight": "source, session_id, timestamp",
    "temperatures": "source, session_id, timestamp",
    "signal_changes": "variant, source, session_id, pin, timestamp",
    "signal_events": "variant, source, session_id",
    "variants": "",
}

RAW_KIND = "surowy"
PREPARED_KIND = "zdarzenia"


def main() -> None:
    tables: dict[str, list[pd.DataFrame]] = {name: [] for name in TABLES}

    for source, name, db_path in raw_databases():
        sessions = load_sessions(db_path)
        tables["sessions"].append(sessions_rows(sessions).assign(source=source, source_name=name))

        for session_id in sessions["session_id"]:
            data = load_session(db_path, int(session_id))
            keys = {"source": source, "session_id": int(session_id)}
            tables["weight"].append(data.weight.pipe(_epoch_seconds, "timestamp").assign(**keys))
            tables["temperatures"].append(data.temperatures.pipe(_epoch_seconds, "timestamp").assign(**keys))
            if data.pins.empty:
                continue
            for variant_name, variant in VARIANTS.items():
                signal_events = detect_events(data.pins, data.end, variant=variant)
                variant_keys = {"variant": variant_name, **keys}
                tables["signal_changes"].append(signal_changes(data.pins, signal_events).assign(**variant_keys))
                tables["signal_events"].append(events_rows(signal_events).assign(**variant_keys))
        print(f"{source}: {db_path.relative_to(DATA_DIR)}")

    tables["variants"].append(variants_rows())
    write({name: pd.concat(frames, ignore_index=True) for name, frames in tables.items()})
    print(f"zapisano {PREPARED_DB.relative_to(PROJECT_DIR)}, warianty: {', '.join(VARIANTS)}")


def sessions_rows(sessions: pd.DataFrame) -> pd.DataFrame:
    table = sessions[
        ["session_id", "started_at", "first_at", "last_at", "weight_count", "pin_count", "temperature_count"]
    ]
    for column in ("started_at", "first_at", "last_at"):
        table = table.pipe(_epoch_seconds, column)
    return table


def variants_rows() -> pd.DataFrame:
    exported_at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    commit = _git("rev-parse", "--short", "HEAD")
    dirty = _git("status", "--porcelain", "--", "src") != ""

    return pd.DataFrame(
        [
            {
                "variant": name,
                "description": variant.description,
                "current": int(name == CURRENT_VARIANT),
                "rules": "; ".join(f"pin {pin} ({role}): {variant.rules(pin)}" for pin, role in PIN_ROLES.items()),
                "default_rules": str(variant.default),
                "commit": commit + ("+zmiany" if dirty else ""),
                "exported_at": exported_at,
            }
            for name, variant in VARIANTS.items()
        ]
    )


def _git(*arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=PROJECT_DIR, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def raw_databases() -> list[tuple[str, str, Path]]:
    config = yaml.safe_load(GRAFANA_DATASOURCES.read_text())
    sources = []

    for datasource in config["datasources"]:
        path = Path(datasource["jsonData"]["path"])
        local = DATA_DIR / path.relative_to(GRAFANA_DATA_DIR)
        if local != PREPARED_DB:
            sources.append((datasource["uid"], datasource["name"], local))

    return sources


def signal_changes(pins: pd.DataFrame, signal_events: SignalEvents) -> pd.DataFrame:
    raw = pins[pins.groupby("pin")["value"].diff() != 0]
    raw_states = (raw["value"] == ACTIVE_LEVEL).map({True: RAW_ACTIVE, False: INACTIVE})

    first_records = pins.groupby("pin", as_index=False)["timestamp"].min()
    marked = pd.concat(
        [
            signal_events.events.assign(state=EVENT),
            signal_events.too_short.assign(state=REJECTED),
        ],
        ignore_index=True,
    )
    starts = marked[["pin", "start", "state"]].rename(columns={"start": "timestamp"})
    ends = marked.loc[~marked["open_end"], ["pin", "end"]].rename(columns={"end": "timestamp"})

    prepared = pd.concat(
        [first_records.assign(state=INACTIVE), starts, ends.assign(state=INACTIVE)],
        ignore_index=True,
    )
    prepared = prepared.drop_duplicates(subset=["pin", "timestamp"], keep="last")

    return pd.concat(
        [
            pd.DataFrame(
                {"pin": raw["pin"], "kind": RAW_KIND, "timestamp": raw["timestamp"], "state": raw_states}
            ),
            prepared.assign(kind=PREPARED_KIND),
        ],
        ignore_index=True,
    ).pipe(_epoch_seconds, "timestamp")


def events_rows(signal_events: SignalEvents) -> pd.DataFrame:
    rows = pd.concat(
        [signal_events.events.assign(rejected=0), signal_events.too_short.assign(rejected=1)],
        ignore_index=True,
    )
    table = pd.DataFrame(
        {
            "pin": rows["pin"],
            "role": rows["role"],
            "start": rows["start"],
            "end": rows["end"],
            "duration": rows["duration"].dt.total_seconds(),
            "pulse_count": rows["pulse_count"],
            "active_share": rows["active_share"],
            "rejected": rows["rejected"],
        }
    )
    return table.pipe(_epoch_seconds, "start").pipe(_epoch_seconds, "end")


def write(tables: dict[str, pd.DataFrame]) -> None:
    PREPARED_DB.parent.mkdir(parents=True, exist_ok=True)
    temporary = PREPARED_DB.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)

    with sqlite3.connect(temporary) as connection:
        for name, table in tables.items():
            table.to_sql(name, connection, index=False)
        for name, columns in TABLES.items():
            if columns:
                connection.execute(f"CREATE INDEX {name}_session ON {name} ({columns})")
    temporary.replace(PREPARED_DB)


def _epoch_seconds(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    epoch = pd.Timestamp(0, tz="UTC")
    return frame.assign(**{column: (frame[column] - epoch).dt.total_seconds()})


if __name__ == "__main__":
    main()
