import sqlite3

import pandas as pd

from dms_analysis.data.paths import temperatures_path
from dms_analysis.data.prepare import source_commit, write_database
from dms_analysis.temperatures.features import MAX_FILLED_GAP_S, WINDOWS_S


def run_row(info: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "period": info["id"],
                "commit": source_commit(),
                "catalog_sha256": info["catalog_sha256"],
                "prepared_at": info["prepared_at"],
                "computed_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                "windows_s": ",".join(str(window) for window in WINDOWS_S),
                "max_filled_gap_s": MAX_FILLED_GAP_S,
            }
        ]
    )


def save(info: pd.Series, frame: pd.DataFrame) -> None:
    target = temperatures_path(str(info["id"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    write_database(target, {"run": run_row(info), "temperatures": frame})


def load(full_id: str) -> tuple[pd.Series, pd.DataFrame]:
    path = temperatures_path(full_id)
    if not path.exists():
        raise FileNotFoundError(f"brak {path} — uruchom: uv run python -m dms_analysis.temperatures compute")

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        run = pd.read_sql_query("SELECT * FROM run", connection).iloc[0]
        frame = pd.read_sql_query("SELECT * FROM temperatures", connection)
    finally:
        connection.close()
    return run, frame
