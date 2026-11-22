import argparse
import sys

import pandas as pd

from dms_analysis.data.catalog import CatalogError, load_catalog
from dms_analysis.data.prepare import is_stale
from dms_analysis.events import store
from dms_analysis.events.values import (
    EXCLUDED_COLLISION,
    EXCLUDED_WINDOW,
    dispense_tables,
    load_period_data,
    refill_table,
    stage_periods,
)


def main() -> int:
    arguments = parser().parse_args()
    try:
        return compute(arguments.period)
    except (CatalogError, FileNotFoundError) as error:
        print(f"błąd: {error}", file=sys.stderr)
        return 1


def parser() -> argparse.ArgumentParser:
    main_parser = argparse.ArgumentParser(prog="python -m dms_analysis.events", description="Etap 2: wartości zdarzeń")
    commands = main_parser.add_subparsers(required=True)
    run = commands.add_parser("compute", help="liczy masy odsypów i zasypów")
    run.add_argument("--period", help="pełny identyfikator okresu (domyślnie wszystkie okresy etapu 2)")
    return main_parser


def compute(only: str | None) -> int:
    periods = stage_periods(load_catalog())
    stale = [period.full_id for period in periods if is_stale(period)]
    if stale:
        print(f"błąd: katalog zmieniony od ostatniego prepare ({', '.join(stale)}) — uruchom prepare", file=sys.stderr)
        return 1

    loaded = [load_period_data(period) for period in periods]
    tables = {data.full_id: dispense_tables(data) for data in loaded}

    for data in loaded:
        if only is not None and data.full_id != only:
            continue
        dispenses, values = tables[data.full_id]
        store.save(
            data,
            {
                "dispenses": dispenses,
                "dispense_values": values,
                "refills": refill_table(data),
            },
        )
        print(f"{data.full_id}: {summary(dispenses)}")
    return 0


def summary(dispenses: pd.DataFrame) -> str:
    counts = dispenses["excluded"].value_counts()
    analysed = int(counts.get("", 0))
    window = int(counts.get(EXCLUDED_WINDOW, 0))
    collision = int(counts.get(EXCLUDED_COLLISION, 0))
    return f"{len(dispenses)} odsypów, do analizy {analysed}, wypada: okno {window}, kolizja {collision}"


if __name__ == "__main__":
    sys.exit(main())
