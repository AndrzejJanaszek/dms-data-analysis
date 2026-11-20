import sqlite3
from pathlib import Path

import pandas as pd

from dms_analysis.data.paths import balance_path, errors_path, errors_summary_path, schedule_path
from dms_analysis.data.prepare import source_commit, write_database
from dms_analysis.errors import fragments, grid, schedule, stats
from dms_analysis.errors.replay import ReplayResult
from dms_analysis.events.values import BASE_DELAY_S, BASE_METHOD, DELAYS_S, MIN_WINDOW_SAMPLES, WINDOW_S, PeriodData

PERIOD_TABLES = ("run", "days", "grid_counts", "grid_skipped", "grid_worst")
SUMMARY_TABLES = ("run", "inputs", "grid_stats")
BALANCE_TABLES = ("run", "balance")
SCHEDULE_TABLES = ("run", "inputs", "schedules", "bases", "pairs", "pair_errors", "replayed", "stats", "ratio")


def parameters() -> dict[str, object]:
    return {
        "commit": source_commit(),
        "computed_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "window_s": WINDOW_S,
        "delays_s": ",".join(f"{delay:g}" for delay in DELAYS_S),
        "durations_s": ",".join(str(duration) for duration in grid.DURATIONS_S),
        "min_window_samples": MIN_WINDOW_SAMPLES,
        "base_method": BASE_METHOD,
        "base_delay_s": BASE_DELAY_S,
        "clearance_s": fragments.CLEARANCE_S,
        "max_sample_step_s": fragments.MAX_SAMPLE_STEP_S,
        "day_start_h": fragments.DAY_START_H,
        "day_end_h": fragments.DAY_END_H,
        "worst_separation_s": grid.WORST_SEPARATION_S,
    }


def period_run(data: PeriodData) -> dict[str, object]:
    return {
        "period": data.full_id,
        "time": data.time,
        "has_temperatures": int(data.has_temperatures),
        "division_kg": data.division_kg,
        "catalog_sha256": data.catalog_sha256,
        "prepared_at": data.prepared_at,
        **parameters(),
    }


def save_period(data: PeriodData, days: pd.DataFrame, result: grid.GridResult) -> None:
    tables = {
        "run": pd.DataFrame([period_run(data)]),
        "days": days,
        "grid_counts": result.counts,
        "grid_skipped": result.skipped,
        "grid_worst": result.worst,
    }
    _write(errors_path(data.full_id), tables)


def load_period(full_id: str) -> dict[str, pd.DataFrame]:
    return _read(errors_path(full_id), PERIOD_TABLES, "grid")


def save_balance(data: PeriodData, table: pd.DataFrame) -> None:
    run = {**period_run(data), "night_window": "0:00–4:00", "day_windows": "4:50–5:00, 22:00–22:10"}
    _write(balance_path(data.full_id), {"run": pd.DataFrame([run]), "balance": table})


def load_balance(full_id: str) -> dict[str, pd.DataFrame]:
    return _read(balance_path(full_id), BALANCE_TABLES, "balance")


def save_summary(site: str, inputs: pd.DataFrame, grid_stats: pd.DataFrame) -> None:
    run = {
        "site": site,
        **parameters(),
        "bootstrap_samples": stats.BOOTSTRAP_SAMPLES,
        "confidence": stats.CONFIDENCE,
        "seed": stats.SEED,
    }
    _write(errors_summary_path(site), {"run": pd.DataFrame([run]), "inputs": inputs, "grid_stats": grid_stats})


def load_summary(site: str) -> dict[str, pd.DataFrame]:
    return _read(errors_summary_path(site), SUMMARY_TABLES, "grid")


def _write(target: Path, tables: dict[str, pd.DataFrame]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    write_database(target, tables)


def _read(path: Path, tables: tuple[str, ...], command: str) -> dict[str, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"brak {path} — uruchom: uv run python -m dms_analysis.errors {command}")

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {name: pd.read_sql_query(f"SELECT * FROM {name}", connection) for name in tables}
    finally:
        connection.close()


def save_schedule(site: str, inputs: pd.DataFrame, result: ReplayResult) -> None:
    run = {
        "site": site,
        **parameters(),
        "stitch_gap_s": schedule.STITCH_GAP_S,
        "stitch_change_kg": schedule.STITCH_CHANGE_KG,
        "bootstrap_samples": stats.BOOTSTRAP_SAMPLES,
        "confidence": stats.CONFIDENCE,
        "seed": stats.SEED,
    }
    tables = {
        "run": pd.DataFrame([run]),
        "inputs": inputs,
        "schedules": result.schedules,
        "bases": result.bases,
        "pairs": result.pairs,
        "pair_errors": result.pair_errors,
        "replayed": result.replayed,
        "stats": result.stats,
        "ratio": result.ratio,
    }
    _write(schedule_path(site), tables)


def load_schedule(site: str) -> dict[str, pd.DataFrame]:
    return _read(schedule_path(site), SCHEDULE_TABLES, "schedule")
