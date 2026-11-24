import argparse
import sqlite3
import sys

import pandas as pd

from dms_analysis.data.catalog import CatalogError, Period, load_catalog
from dms_analysis.data.paths import prepared_path
from dms_analysis.data.prepare import is_stale
from dms_analysis.temperatures import store
from dms_analysis.temperatures.features import features


def main() -> int:
    arguments = parser().parse_args()
    try:
        return compute(arguments.period)
    except (CatalogError, FileNotFoundError) as error:
        print(f"błąd: {error}", file=sys.stderr)
        return 1


def parser() -> argparse.ArgumentParser:
    main_parser = argparse.ArgumentParser(
        prog="python -m dms_analysis.temperatures", description="Etap 5: dane pochodne temperatur"
    )
    commands = main_parser.add_subparsers(required=True)
    run = commands.add_parser("compute", help="liczy wielkości z temperatur na siatce 1 s")
    run.add_argument("--period", help="pełny identyfikator okresu (domyślnie wszystkie okresy z temperaturami)")
    return main_parser


def compute(only: str | None) -> int:
    periods = [period for period in temperature_periods() if only is None or period.full_id == only]
    if only is not None and not periods:
        print(f"błąd: {only} nie jest okresem z temperaturami (kanał temperatures = ok)", file=sys.stderr)
        return 1

    stale = [period.full_id for period in periods if is_stale(period)]
    if stale:
        print(f"błąd: katalog zmieniony od ostatniego prepare ({', '.join(stale)}) — uruchom prepare", file=sys.stderr)
        return 1

    for period in periods:
        info, temperatures = read_prepared(period)
        frame = features(temperatures, period.hardware)
        store.save(info, frame)
        numbers = [sensor.number for sensor in period.hardware.temperature_sensors]
        print(f"{period.full_id}: {describe(frame, numbers)}")
    return 0


def temperature_periods() -> list[Period]:
    return [period for period in load_catalog().periods.values() if period.channels["temperatures"] == "ok"]


def read_prepared(period: Period) -> tuple[pd.Series, pd.DataFrame]:
    connection = sqlite3.connect(f"file:{prepared_path(period.full_id)}?mode=ro", uri=True)
    try:
        info = pd.read_sql_query("SELECT * FROM period", connection).iloc[0]
        temperatures = pd.read_sql_query("SELECT timestamp, sensor, value FROM temperatures", connection)
    finally:
        connection.close()
    return info, temperatures


def describe(frame: pd.DataFrame, numbers: list[int]) -> str:
    filled = sum(int(frame[f"filled_{number}"].sum()) for number in numbers)
    complete = int(frame["mean"].notna().sum())
    return f"{len(frame)} s, pełne 4 czujniki {complete} s, uzupełnione odczyty {filled}"


if __name__ == "__main__":
    sys.exit(main())
