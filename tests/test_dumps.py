import sqlite3
from datetime import date

from dms_analysis.data.dumps import add_dump, sha256


def test_add_dump_merges_non_empty_wal(tmp_path):
    source_dir = tmp_path / "rpi"
    source_dir.mkdir()
    source = source_dir / "measurement-data-saver.db"
    writer = sqlite3.connect(source)
    writer.execute("PRAGMA journal_mode = WAL")
    writer.execute("PRAGMA wal_autocheckpoint = 0")
    writer.execute("CREATE TABLE measurements (id INTEGER PRIMARY KEY, value REAL)")
    writer.executemany("INSERT INTO measurements (value) VALUES (?)", [(value,) for value in range(100)])
    writer.commit()

    wal = source.with_name(source.name + "-wal")
    assert wal.stat().st_size > 0
    source_sum = sha256(source)

    block = add_dump(source, "rpi4", date(2026, 11, 9), raw_dir=tmp_path / "raw")
    assert sha256(source) == source_sum
    writer.close()

    target = tmp_path / "raw" / "rpi4" / "2026-11-09" / "measurement-data-saver.db"
    copy = sqlite3.connect(f"file:{target}?mode=ro&immutable=1", uri=True)
    assert copy.execute("SELECT COUNT(*) FROM measurements").fetchone()[0] == 100
    copy.close()

    assert "scalono niepusty plik -wal" in block
    assert f'sha256 = "{sha256(target)}"' in block
    assert not target.with_name(target.name + "-wal").exists()
