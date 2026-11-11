import sqlite3
from dataclasses import dataclass

import pandas as pd

from dms_analysis.data import check
from dms_analysis.data.paths import prepared_path, results_path
from dms_analysis.data.prepare import source_commit, write_database
from dms_analysis.events import synthetic
from dms_analysis.events.values import BASE_DELAY_S, BASE_METHOD, DELAYS_S, WINDOW_S, PeriodData

TABLES = ("run", "dispenses", "dispense_values", "refills", "synthetic_trials")


@dataclass(frozen=True)
class Results:
    full_id: str
    run: pd.Series
    dispenses: pd.DataFrame
    values: pd.DataFrame
    refills: pd.DataFrame
    trials: pd.DataFrame


def run_row(data: PeriodData) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "period": data.full_id,
                "time": data.time,
                "has_temperatures": int(data.has_temperatures),
                "division_kg": data.division_kg,
                "commit": source_commit(),
                "catalog_sha256": data.catalog_sha256,
                "prepared_at": data.prepared_at,
                "computed_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                "window_s": WINDOW_S,
                "delays_s": ",".join(f"{delay:g}" for delay in DELAYS_S),
                "base_method": BASE_METHOD,
                "base_delay_s": BASE_DELAY_S,
                "day_start_h": synthetic.DAY_START_H,
                "day_end_h": synthetic.DAY_END_H,
                "trials_per_day_part": synthetic.TRIALS_PER_DAY_PART,
                "clearance_s": synthetic.CLEARANCE_S,
                "mass_jitter_kg": data.division_kg / 2,
                "seed": synthetic.SEED,
            }
        ]
    )


def save(data: PeriodData, tables: dict[str, pd.DataFrame]) -> None:
    target = results_path(data.full_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_database(target, {"run": run_row(data), **tables})


def load(full_id: str) -> Results:
    path = results_path(full_id)
    if not path.exists():
        raise FileNotFoundError(f"brak {path} — uruchom: uv run python -m dms_analysis.events compute")

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        frames = {name: pd.read_sql_query(f"SELECT * FROM {name}", connection) for name in TABLES}
    finally:
        connection.close()

    return Results(
        full_id=full_id,
        run=frames["run"].iloc[0],
        dispenses=_local(frames["dispenses"], "start", "end"),
        values=frames["dispense_values"].assign(no_drop=lambda frame: frame["no_drop"].astype(bool)),
        refills=_local(frames["refills"], "start", "end"),
        trials=_local(frames["synthetic_trials"], "start"),
    )


def _local(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    return frame.assign(
        **{column: pd.to_datetime(frame[column], unit="s", utc=True).dt.tz_convert(check.TIMEZONE) for column in columns}
    )


def staleness(results: Results) -> str | None:
    prepared = prepared_path(results.full_id)
    connection = sqlite3.connect(f"file:{prepared}?mode=ro", uri=True)
    try:
        prepared_at = connection.execute("SELECT prepared_at FROM period").fetchone()[0]
    finally:
        connection.close()
    if prepared_at != results.run["prepared_at"]:
        return f"{results.full_id}: przygotowana baza nowsza niż wyniki — uruchom ponownie compute"
    return None
