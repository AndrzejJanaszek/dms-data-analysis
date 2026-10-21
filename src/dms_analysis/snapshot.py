import hashlib
import lzma
import shutil
import sqlite3
from pathlib import Path

import pandas as pd

from dms_analysis.export import DATA_DIR, PREPARED_DB, PROJECT_DIR, TABLES

ANALYSIS_SOURCE = "sqlite-2026-10-18"
ANALYSIS_SESSION = 9
SNAPSHOT_DIR = DATA_DIR / "snapshots"
SNAPSHOT_NAME = "prepared-obiekt-sesja-9"


def main() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    database = SNAPSHOT_DIR / f"{SNAPSHOT_NAME}.db"
    archive = SNAPSHOT_DIR / f"{SNAPSHOT_NAME}.db.xz"
    database.unlink(missing_ok=True)

    with sqlite3.connect(database) as connection:
        connection.execute("ATTACH DATABASE ? AS prepared", (f"file:{PREPARED_DB}?mode=ro",))
        for name, columns in TABLES.items():
            condition = "" if name == "variants" else " WHERE source = ? AND session_id = ?"
            parameters = () if name == "variants" else (ANALYSIS_SOURCE, ANALYSIS_SESSION)
            connection.execute(f"CREATE TABLE {name} AS SELECT * FROM prepared.{name}{condition}", parameters)
            if columns:
                connection.execute(f"CREATE INDEX {name}_session ON {name} ({columns})")
        connection.commit()
        connection.execute("DETACH DATABASE prepared")
        summary = _summary(connection)
    _vacuum(database)

    with database.open("rb") as source, lzma.open(archive, "wb", preset=6) as target:
        shutil.copyfileobj(source, target)

    print(f"zapisano {archive.relative_to(PROJECT_DIR)} ({archive.stat().st_size / 1e6:.1f} MB)")
    print(f"rozpakowany {database.stat().st_size / 1e6:.1f} MB, SHA-256 {_sha256(database)}")
    print(summary)
    database.unlink()


def _summary(connection: sqlite3.Connection) -> str:
    period = pd.read_sql_query("SELECT first_at, last_at FROM sessions", connection)
    first, last = (
        pd.to_datetime(period[column], unit="s", utc=True).dt.tz_convert("Europe/Warsaw").iloc[0]
        for column in ("first_at", "last_at")
    )
    counts = {name: connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] for name in TABLES}
    commit = connection.execute('SELECT DISTINCT "commit" FROM variants').fetchone()[0]
    return f"okres {first:%Y-%m-%d %H:%M} – {last:%Y-%m-%d %H:%M}, commit eksportu {commit}, wiersze {counts}"


def _vacuum(database: Path) -> None:
    connection = sqlite3.connect(database)
    connection.execute("VACUUM")
    connection.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
