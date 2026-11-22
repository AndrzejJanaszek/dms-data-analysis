import argparse
import sys
from collections import defaultdict

import pandas as pd

from dms_analysis.data.catalog import CatalogError, Period, load_catalog
from dms_analysis.data.prepare import is_stale
from dms_analysis.data.review import load_marks
from dms_analysis.errors import store
from dms_analysis.errors.balance import VARIANTS, period_balance
from dms_analysis.errors.replay import compute as replay_compute
from dms_analysis.errors.schedule import load as load_replay_period
from dms_analysis.errors.fragments import days_table, free_intervals, real_events
from dms_analysis.errors.grid import period_grid
from dms_analysis.errors.summary import site_stats
from dms_analysis.events.values import load_period_data, stage_periods

REAL_TIME = "real"


def main() -> int:
    arguments = parser().parse_args()
    try:
        return arguments.command(arguments.period)
    except (CatalogError, FileNotFoundError) as error:
        print(f"błąd: {error}", file=sys.stderr)
        return 1


def parser() -> argparse.ArgumentParser:
    main_parser = argparse.ArgumentParser(prog="python -m dms_analysis.errors", description="Etap 3: miary błędu")
    commands = main_parser.add_subparsers(required=True)
    run = commands.add_parser("grid", help="test na siatce: błąd odsypu bez masy dla każdej sekundy startu")
    run.add_argument("--period", help="pełny identyfikator okresu (domyślnie wszystkie okresy etapu 2)")
    run.set_defaults(command=grid_command)
    balance = commands.add_parser("balance", help="bilans masy doby: noc–noc i 5–22")
    balance.add_argument("--period", help="pełny identyfikator okresu (domyślnie wszystkie okresy etapu 2)")
    balance.set_defaults(command=balance_command)
    replay = commands.add_parser("schedule", help="prawdziwy harmonogram odtworzony na spokojnych dobach")
    replay.set_defaults(command=schedule_command, period=None)
    return main_parser


def fresh_periods() -> list[Period] | None:
    periods = stage_periods(load_catalog())
    stale = [period.full_id for period in periods if is_stale(period)]
    if stale:
        print(f"błąd: katalog zmieniony od ostatniego prepare ({', '.join(stale)}) — uruchom prepare", file=sys.stderr)
        return None
    return periods


def grid_command(only: str | None) -> int:
    periods = fresh_periods()
    if periods is None:
        return 1

    marks = load_marks()
    for period in periods:
        if only is None or period.full_id == only:
            compute_period(period, marks)

    for site, site_periods in by_site(periods).items():
        summarize(site, site_periods)
    return 0


def compute_period(period: Period, marks: pd.DataFrame) -> None:
    data = load_period_data(period)
    events = real_events(data, marks)
    free = free_intervals(data, events)
    result = period_grid(data.weight, free)
    store.save_period(data, days_table(data, events, marks), result)

    used = int(result.counts.groupby(["method", "delay_s", "duration_s"])["starts"].sum().max())
    skipped = int(result.skipped.groupby("duration_s")["starts"].sum().max()) if not result.skipped.empty else 0
    print(
        f"{data.full_id}: {len(free)} fragmentów bez zdarzeń, do {used} startów na T, pominięte (< 3 próbki): do {skipped} na T"
    )


def balance_command(only: str | None) -> int:
    periods = fresh_periods()
    if periods is None:
        return 1

    marks = load_marks()
    for period in periods:
        if only is not None and period.full_id != only:
            continue
        data = load_period_data(period)
        table = period_balance(data, real_events(data, marks), marks)
        store.save_balance(data, table)
        valid = table[table["valid"] == 1]
        counts = ", ".join(f"{variant}: {int((valid['variant'] == variant).sum())}" for variant in VARIANTS)
        print(f"{data.full_id}: ważne doby — {counts}")
    return 0


def schedule_command(_: str | None) -> int:
    periods = fresh_periods()
    if periods is None:
        return 1

    marks = load_marks()
    for site, site_periods in by_site(periods).items():
        real = [period for period in site_periods if period.time == REAL_TIME]
        loaded = [load_replay_period(load_period_data(period), marks) for period in real]
        result = replay_compute(loaded)
        inputs = pd.DataFrame(
            [
                {"period": item.data.full_id, "time": item.data.time, "prepared_at": item.data.prepared_at}
                for item in loaded
            ]
        )
        store.save_schedule(site, inputs, result)
        valid = int(result.pairs["valid"].sum()) if not result.pairs.empty else 0
        print(
            f"{site}: {len(result.schedules)} harmonogramów, {len(result.bases)} dób bazowych, ważne pary {valid}/{len(result.pairs)}"
        )
    return 0


def by_site(periods: list[Period]) -> dict[str, list[Period]]:
    grouped = defaultdict(list)
    for period in periods:
        grouped[period.site].append(period)
    return dict(grouped)


def summarize(site: str, periods: list[Period]) -> None:
    loaded = {period.full_id: store.load_period(period.full_id) for period in periods}
    counts = {full_id: tables["grid_counts"] for full_id, tables in loaded.items()}
    real_time = [period.full_id for period in periods if period.time == REAL_TIME]
    inputs = pd.concat([tables["run"][["period", "time", "prepared_at", "commit"]] for tables in loaded.values()])
    store.save_summary(site, inputs, site_stats(counts, real_time))
    print(f"{site}: zestawienie z {len(periods)} okresów (czas rzeczywisty: {len(real_time)})")


if __name__ == "__main__":
    sys.exit(main())
