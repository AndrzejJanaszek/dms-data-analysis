import argparse
import sys
from datetime import date
from pathlib import Path

from dms_analysis.data.catalog import Catalog, CatalogError, Period, load_catalog
from dms_analysis.data.dumps import add_dump, verify
from dms_analysis.data.extract import extract
from dms_analysis.data.paths import PROJECT_DIR, prepared_path
from dms_analysis.data.prepare import build_all_periods, prepare
from dms_analysis.data.sessions import period_draft, sessions_for_dump, sessions_table, uncatalogued


def main() -> int:
    arguments = parser().parse_args()
    try:
        return arguments.command(arguments)
    except (CatalogError, FileNotFoundError, FileExistsError) as error:
        print(f"błąd: {error}", file=sys.stderr)
        return 1


def parser() -> argparse.ArgumentParser:
    main_parser = argparse.ArgumentParser(
        prog="python -m dms_analysis.data", description="Katalog danych i przygotowanie okresów"
    )
    commands = main_parser.add_subparsers(required=True)

    add = commands.add_parser("add-dump", help="kopiuje surowy zrzut do data/raw/ i wypisuje blok [[dump]]")
    add.add_argument("file", type=Path)
    add.add_argument("--device", required=True)
    add.add_argument("--pulled", required=True, type=date.fromisoformat, help="data zgrania RRRR-MM-DD")
    add.add_argument("--id", dest="dump_id", help="identyfikator zrzutu (domyślnie <urządzenie>/<data>)")
    add.add_argument("--program", default="measurement-data-saver")
    add.add_argument("--note", default="")
    add.set_defaults(command=run_add_dump)

    check = commands.add_parser("verify", help="sprawdza sumy SHA-256 zrzutów z raw.toml")
    check.set_defaults(command=run_verify)

    listing = commands.add_parser("sessions", help="sesje zrzutu spoza katalogu i szkic bloku [[period]]")
    listing.add_argument("dump")
    listing.add_argument("--site", help="obiekt szkicu")
    listing.add_argument("--session", type=int, help="sesja (albo odcinek starej bazy) szkicu")
    listing.add_argument("--all", action="store_true", help="także sesje już w katalogu")
    listing.set_defaults(command=run_sessions)

    slices = commands.add_parser("extract", help="tworzy brakujące surowe wycinki okresów")
    slices.add_argument("--period", help="pełny identyfikator okresu, np. swietochlowice-1/0002-2026-09-28")
    slices.set_defaults(command=run_extract)

    build = commands.add_parser("prepare", help="buduje przygotowane bazy okresów i periods.db dla Grafany")
    build.add_argument("--period", help="pełny identyfikator okresu")
    build.set_defaults(command=run_prepare)

    return main_parser


def run_add_dump(arguments: argparse.Namespace) -> int:
    block = add_dump(
        arguments.file,
        arguments.device,
        arguments.pulled,
        dump_id=arguments.dump_id,
        program=arguments.program,
        note=arguments.note,
    )
    print("Dopisz do catalog/raw.toml:\n")
    print(block)
    return 0


def run_verify(arguments: argparse.Namespace) -> int:
    matching, missing, different = verify()
    for dump_id in matching:
        print(f"OK          {dump_id}")
    for line in missing:
        print(f"BRAK        {line} (pobierz z Drive: INZYNIERKA/dms-data/raw/)")
    for line in different:
        print(f"NIEZGODNY   {line}")
    return 1 if different else 0


def run_sessions(arguments: argparse.Namespace) -> int:
    catalog = load_catalog()
    sessions = sessions_for_dump(catalog, arguments.dump)
    shown = sessions if arguments.all else uncatalogued(catalog, arguments.dump, sessions)
    print(sessions_table(shown).to_string(index=False))

    if arguments.session is not None:
        if not arguments.site:
            raise CatalogError("do szkicu podaj też --site")
        print("\nSzkic do wklejenia do catalog/" + arguments.site + ".toml:\n")
        print(period_draft(catalog, arguments.dump, sessions, arguments.site, arguments.session))
    return 0


def run_extract(arguments: argparse.Namespace) -> int:
    catalog = load_catalog()
    for period in selected(catalog, arguments.period):
        created = extract(catalog, period)
        if created is None:
            print(f"jest już  {period.full_id}")
        else:
            print(f"utworzono {created.relative_to(PROJECT_DIR)} ({created.stat().st_size / 1e6:.1f} MB)")
    return 0


def run_prepare(arguments: argparse.Namespace) -> int:
    catalog = load_catalog()
    for period in selected(catalog, arguments.period):
        warnings = prepare(catalog, period)
        print(f"przygotowano {prepared_path(period.full_id).relative_to(PROJECT_DIR)}")
        for warning in warnings:
            print(f"  uwaga: {warning}")
    included = build_all_periods(catalog)
    print(f"periods.db (wszystkie okresy, dla Grafany): {len(included)}")
    return 0


def selected(catalog: Catalog, full_id: str | None) -> list[Period]:
    return list(catalog.periods.values()) if full_id is None else [catalog.period(full_id)]


if __name__ == "__main__":
    sys.exit(main())
