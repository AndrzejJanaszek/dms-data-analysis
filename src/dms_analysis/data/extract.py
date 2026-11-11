import lzma
import shutil
import sqlite3
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from dms_analysis.data.catalog import Catalog, Period
from dms_analysis.data.dumps import dump_path, read_only_uri
from dms_analysis.data.paths import raw_period_path

READ_ONLY = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH


def extract(catalog: Catalog, period: Period) -> Path | None:
    target = raw_period_path(period.full_id)
    if target.exists():
        return None

    dump = catalog.dumps[period.source.dump]
    source = dump_path(dump)
    if not source.exists():
        raise FileNotFoundError(f"brak zrzutu {source} (pobierz go z Drive)")

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as working:
        database = Path(working) / "period.db"
        copy_rows(source, database, period)
        compressed = Path(working) / target.name
        with database.open("rb") as raw, lzma.open(compressed, "wb", preset=9) as packed:
            shutil.copyfileobj(raw, packed)
        compressed.replace(target)
    target.chmod(READ_ONLY)
    return target


def copy_rows(source: Path, database: Path, period: Period) -> None:
    connection = sqlite3.connect(database)
    connection.execute("ATTACH DATABASE ? AS source", (read_only_uri(source),))
    schema = connection.execute(
        "SELECT type, name, sql FROM source.sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
        " ORDER BY type = 'index'"
    ).fetchall()

    for kind, name, sql in schema:
        connection.execute(sql)
        if kind == "table":
            condition, parameters = _rows_condition(connection, name, period)
            connection.execute(f"INSERT INTO main.{name} SELECT * FROM source.{name} WHERE {condition}", parameters)
    connection.commit()
    connection.execute("DETACH DATABASE source")
    connection.execute("VACUUM")
    connection.close()


def _rows_condition(connection: sqlite3.Connection, table: str, period: Period) -> tuple[str, tuple[int, ...]]:
    columns = {row[1] for row in connection.execute(f"PRAGMA source.table_info({table})")}
    source = period.source

    if source.session is not None:
        if table == "sessions":
            return "id = ?", (source.session,)
        if "session_id" in columns:
            return "session_id = ?", (source.session,)
        raise ValueError(f"tabela {table} bez session_id — nie wiem, które wiersze należą do sesji")

    if source.first_id is None or source.last_id is None:
        raise ValueError("okres bez sesji i bez zakresu id")
    return "id BETWEEN ? AND ?", (source.first_id, source.last_id)


@contextmanager
def unpacked(full_id: str, directory: Path) -> Iterator[Path]:
    archive = raw_period_path(full_id)
    if not archive.exists():
        raise FileNotFoundError(f"brak wycinka {archive} — uruchom extract")
    with tempfile.TemporaryDirectory(dir=directory) as working:
        database = Path(working) / "period.db"
        with lzma.open(archive, "rb") as packed, database.open("wb") as raw:
            shutil.copyfileobj(packed, raw)
        yield database
