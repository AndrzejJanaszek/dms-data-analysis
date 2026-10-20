import sqlite3
from pathlib import Path

import pandas as pd
import yaml

from dms_analysis.raw import load_session, load_sessions
from dms_analysis.signals import ACTIVE_LEVEL, SignalEvents, detect_events

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
GRAFANA_DATASOURCES = PROJECT_DIR / "grafana" / "provisioning" / "datasources" / "sqlite.yaml"
GRAFANA_DATA_DIR = "/data"
PREPARED_DB = DATA_DIR / "prepared" / "signals.db"

INACTIVE = 0
RAW_ACTIVE = 1
EVENT = 2
REJECTED = 3

RAW_KIND = "surowy"
PREPARED_KIND = "zdarzenia"


def main() -> None:
    changes = []
    events = []

    for source, db_path in raw_databases():
        for session_id in load_sessions(db_path)["session_id"]:
            data = load_session(db_path, int(session_id))
            if data.pins.empty:
                continue
            signal_events = detect_events(data.pins, data.end)
            keys = {"source": source, "session_id": int(session_id)}
            changes.append(signal_changes(data.pins, signal_events).assign(**keys))
            events.append(events_rows(signal_events).assign(**keys))
        print(f"{source}: {db_path.relative_to(DATA_DIR)}")

    write(pd.concat(changes, ignore_index=True), pd.concat(events, ignore_index=True))
    print(f"zapisano {PREPARED_DB.relative_to(PROJECT_DIR)}")


def raw_databases() -> list[tuple[str, Path]]:
    config = yaml.safe_load(GRAFANA_DATASOURCES.read_text())
    sources = []

    for datasource in config["datasources"]:
        path = Path(datasource["jsonData"]["path"])
        local = DATA_DIR / path.relative_to(GRAFANA_DATA_DIR)
        if local != PREPARED_DB:
            sources.append((datasource["uid"], local))

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


def write(changes: pd.DataFrame, events: pd.DataFrame) -> None:
    PREPARED_DB.parent.mkdir(parents=True, exist_ok=True)
    temporary = PREPARED_DB.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)

    with sqlite3.connect(temporary) as connection:
        changes.to_sql("signal_changes", connection, index=False)
        events.to_sql("signal_events", connection, index=False)
        connection.execute("CREATE INDEX signal_changes_session ON signal_changes (source, session_id, pin, timestamp)")
        connection.execute("CREATE INDEX signal_events_session ON signal_events (source, session_id)")
    temporary.replace(PREPARED_DB)


def _epoch_seconds(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    epoch = pd.Timestamp(0, tz="UTC")
    return frame.assign(**{column: (frame[column] - epoch).dt.total_seconds()})


if __name__ == "__main__":
    main()
