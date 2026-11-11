import hashlib
import shutil
import sqlite3
import tempfile
from datetime import date
from pathlib import Path

from dms_analysis.data.catalog import RAW_FILE, Dump, load_dumps
from dms_analysis.data.paths import CATALOG_DIR, RAW_DIR


def connect_read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(read_only_uri(path), uri=True)


def read_only_uri(path: Path) -> str:
    return f"file:{path.resolve()}?mode=ro&immutable=1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_path(dump: Dump, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / dump.file


def add_dump(
    source: Path,
    device: str,
    pulled: date,
    dump_id: str | None = None,
    program: str = "measurement-data-saver",
    note: str = "",
    raw_dir: Path = RAW_DIR,
) -> str:
    dump_id = dump_id or f"{device}/{pulled.isoformat()}"
    relative = Path(device) / pulled.isoformat() / source.name
    target = raw_dir / relative
    if target.exists():
        raise FileExistsError(f"{target} już istnieje — zrzutu nie nadpisuję")

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as working:
        copy = Path(working) / source.name
        shutil.copy2(source, copy)
        merged = _merge_wal(source, copy)
        copy.replace(target)

    lines = [
        "[[dump]]",
        f'id = "{dump_id}"',
        f'device = "{device}"',
        f"pulled = {pulled.isoformat()}",
        f'file = "{relative.as_posix()}"',
        f'sha256 = "{sha256(target)}"',
        f'program = "{program}"',
        f'note = "{note}"',
        'contained_in = ""',
    ]
    if merged:
        lines.insert(0, "# scalono niepusty plik -wal z bazą w kopii")
    return "\n".join(lines)


def _merge_wal(source: Path, copy: Path) -> bool:
    wal = source.with_name(source.name + "-wal")
    if not wal.exists() or wal.stat().st_size == 0:
        return False

    shutil.copy2(wal, copy.with_name(copy.name + "-wal"))
    connection = sqlite3.connect(copy)
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.execute("PRAGMA journal_mode = DELETE")
    connection.close()
    for suffix in ("-wal", "-shm"):
        copy.with_name(copy.name + suffix).unlink(missing_ok=True)
    return True


def verify(catalog_dir: Path = CATALOG_DIR, raw_dir: Path = RAW_DIR) -> tuple[list[str], list[str], list[str]]:
    matching, missing, different = [], [], []

    for dump in load_dumps(catalog_dir / RAW_FILE).values():
        path = dump_path(dump, raw_dir)
        if not path.exists():
            missing.append(f"{dump.id}: brak {path.relative_to(raw_dir.parent)}")
        elif sha256(path) != dump.sha256:
            different.append(f"{dump.id}: suma SHA-256 niezgodna z {RAW_FILE}")
        else:
            matching.append(dump.id)
        for suffix in ("-wal", "-shm"):
            if path.with_name(path.name + suffix).exists():
                different.append(f"{dump.id}: obok zrzutu leży plik {suffix} (ktoś otworzył bazę do zapisu)")

    return matching, missing, different
