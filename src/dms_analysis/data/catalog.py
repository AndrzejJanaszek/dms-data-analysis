import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from dms_analysis.data.paths import CATALOG_DIR
from dms_analysis.signals import DISPENSE_PIN, PIN_ROLES, REFILL_PIN

RAW_FILE = "raw.toml"

CHANNELS = ("weight", "dispense", "refill", "temperatures")
SIGNAL_CHANNELS = ("dispense", "refill")
CHANNEL_STATES = ("ok", "flickering", "broken", "not_connected", "no_values")
TIME_STATUSES = ("real", "estimated", "unknown")
ROLE_NAMES = {"dispense": PIN_ROLES[DISPENSE_PIN], "refill": PIN_ROLES[REFILL_PIN]}
CANONICAL_PINS = {"dispense": DISPENSE_PIN, "refill": REFILL_PIN}

PERIOD_ID = re.compile(r"^\d{4}-(\d{4}-\d{2}-\d{2}|unknown)$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CatalogError(Exception):
    pass


@dataclass(frozen=True)
class Dump:
    id: str
    device: str
    pulled: date
    file: str
    sha256: str
    program: str
    note: str
    contained_in: str


@dataclass(frozen=True)
class TemperatureSensor:
    number: int
    address: str
    position: str


@dataclass(frozen=True)
class Hardware:
    id: str
    silo: str
    meter: str
    meter_division_kg: float
    pins: dict[int, str]
    temperature_sensors: tuple[TemperatureSensor, ...]

    def pin_for(self, channel: str) -> int | None:
        matching = [pin for pin, role in self.pins.items() if role == channel]
        return matching[0] if matching else None


@dataclass(frozen=True)
class Site:
    id: str
    name: str
    description: str
    hardware: dict[str, Hardware]


@dataclass(frozen=True)
class Source:
    dump: str
    session: int | None
    first_id: int | None
    last_id: int | None


@dataclass(frozen=True)
class Issue:
    channel: str
    kind: str
    start: datetime
    end: datetime
    note: str


@dataclass(frozen=True)
class ManualEvent:
    channel: str
    start: datetime
    end: datetime | None


@dataclass(frozen=True)
class Period:
    site: str
    id: str
    legacy_name: str
    hardware: Hardware
    source: Source
    time: str
    clock_offset_h: float
    time_uncertainty_h: float
    start: datetime
    end: datetime
    checked: bool
    note: str
    channels: dict[str, str]
    issues: tuple[Issue, ...]
    manual_events: tuple[ManualEvent, ...]
    catalog_sha256: str

    @property
    def full_id(self) -> str:
        return f"{self.site}/{self.id}"

    @property
    def number(self) -> int:
        return int(self.id[:4])


@dataclass(frozen=True)
class Catalog:
    dumps: dict[str, Dump]
    sites: dict[str, Site]
    periods: dict[str, Period]

    def period(self, full_id: str) -> Period:
        if full_id not in self.periods:
            known = ", ".join(self.periods)
            raise CatalogError(f"nie ma okresu {full_id} w katalogu (są: {known})")
        return self.periods[full_id]

    def root_dump(self, dump_id: str) -> str:
        dump = self.dumps[dump_id]
        return dump.contained_in or dump.id


def load_catalog(directory: Path = CATALOG_DIR) -> Catalog:
    dumps = load_dumps(directory / RAW_FILE)
    sites: dict[str, Site] = {}
    periods: dict[str, Period] = {}

    for path in sorted(directory.glob("*.toml")):
        if path.name == RAW_FILE:
            continue
        site, site_periods = _load_site(path, dumps)
        sites[site.id] = site
        for period in site_periods:
            periods[period.full_id] = period

    return Catalog(dumps=dumps, sites=sites, periods=periods)


def load_dumps(path: Path) -> dict[str, Dump]:
    content = _read(path)
    dumps: dict[str, Dump] = {}

    for number, block in enumerate(content.get("dump", []), start=1):
        context = f"{path.name}, zrzut nr {number}"
        dump = Dump(
            id=_text(block, "id", context),
            device=_text(block, "device", context),
            pulled=_date(block, "pulled", context),
            file=_text(block, "file", context),
            sha256=_text(block, "sha256", context),
            program=_text(block, "program", context),
            note=_text(block, "note", context, ""),
            contained_in=_text(block, "contained_in", context, ""),
        )
        _check(dump.id not in dumps, f"{context}: powtórzony identyfikator zrzutu {dump.id}")
        _check(SHA256.match(dump.sha256) is not None, f"{context}: sha256 to nie 64 znaki szesnastkowe")
        dumps[dump.id] = dump

    for dump in dumps.values():
        _check(
            not dump.contained_in or dump.contained_in in dumps,
            f"{path.name}, zrzut {dump.id}: contained_in wskazuje nieistniejący zrzut {dump.contained_in}",
        )
    return dumps


def _load_site(path: Path, dumps: dict[str, Dump]) -> tuple[Site, list[Period]]:
    content = _read(path)
    site_block = _table(content, "site", path.name)
    site_id = _text(site_block, "id", f"{path.name}, [site]")
    _check(site_id == path.stem, f"{path.name}: [site] id = {site_id}, a plik nazywa się {path.stem}")

    hardware_blocks = {}
    for number, block in enumerate(content.get("hardware", []), start=1):
        context = f"{path.name}, [[hardware]] nr {number}"
        hardware_id = _text(block, "id", context)
        _check(hardware_id not in hardware_blocks, f"{context}: powtórzony identyfikator sprzętu {hardware_id}")
        hardware_blocks[hardware_id] = block

    site = Site(
        id=site_id,
        name=_text(site_block, "name", f"{path.name}, [site]"),
        description=_text(site_block, "description", f"{path.name}, [site]", ""),
        hardware={
            hardware_id: _hardware(block, f"{path.name}, sprzęt {hardware_id}")
            for hardware_id, block in hardware_blocks.items()
        },
    )

    periods: list[Period] = []
    for number, block in enumerate(content.get("period", []), start=1):
        context = f"{path.name}, [[period]] nr {number}"
        period = _period(block, site, hardware_blocks, dumps, context)
        _check(
            all(existing.id != period.id for existing in periods),
            f"{context}: powtórzony identyfikator okresu {period.id}",
        )
        _check(
            all(existing.number != period.number for existing in periods),
            f"{context}: powtórzony numer okresu {period.id[:4]}",
        )
        periods.append(period)

    return site, periods


def _hardware(block: dict[str, Any], context: str) -> Hardware:
    pins_block = block.get("pins", {})
    _check(isinstance(pins_block, dict), f"{context}: pins musi być tabelą {{ numer = rola }}")
    pins = {}
    for key, role in pins_block.items():
        _check(key.isdigit(), f"{context}: numer pinu „{key}” nie jest liczbą")
        _check(
            role in SIGNAL_CHANNELS,
            f"{context}: nieznana rola pinu {key}: „{role}” (dozwolone: {_listed(SIGNAL_CHANNELS)})",
        )
        pins[int(key)] = role
    _check(len(set(pins.values())) == len(pins), f"{context}: ta sama rola przypisana do kilku pinów")

    sensors = []
    for number, sensor in enumerate(block.get("temperature_sensors", []), start=1):
        sensor_context = f"{context}, czujnik nr {number}"
        sensors.append(
            TemperatureSensor(
                number=_integer(sensor, "number", sensor_context),
                address=_text(sensor, "address", sensor_context),
                position=_text(sensor, "position", sensor_context),
            )
        )
    _check(len({sensor.address for sensor in sensors}) == len(sensors), f"{context}: powtórzony adres czujnika")
    _check(len({sensor.number for sensor in sensors}) == len(sensors), f"{context}: powtórzony numer czujnika")

    return Hardware(
        id=_text(block, "id", context),
        silo=_text(block, "silo", context),
        meter=_text(block, "meter", context),
        meter_division_kg=_number(block, "meter_division_kg", context),
        pins=pins,
        temperature_sensors=tuple(sensors),
    )


def _period(
    block: dict[str, Any],
    site: Site,
    hardware_blocks: dict[str, dict[str, Any]],
    dumps: dict[str, Dump],
    context: str,
) -> Period:
    period_id = _text(block, "id", context)
    context = f"{context} ({period_id})"
    _check(
        PERIOD_ID.match(period_id) is not None,
        f"{context}: identyfikator ma mieć postać NNNN-RRRR-MM-DD albo NNNN-unknown",
    )

    hardware_id = _text(block, "hardware", context)
    _check(hardware_id in site.hardware, f"{context}: hardware wskazuje nieistniejący sprzęt {hardware_id}")
    hardware = site.hardware[hardware_id]

    time_status = _choice(block, "time", TIME_STATUSES, context)
    clock_offset_h = _number(block, "clock_offset_h", context)
    _check(
        time_status != "unknown" or clock_offset_h == 0,
        f'{context}: przy time = "unknown" clock_offset_h musi być 0 (czasy to czas zegara urządzenia)',
    )
    start = _datetime(block, "start", context)
    end = _datetime(block, "end", context)
    _check(start < end, f"{context}: start nie jest przed end")

    channels_block = _table(block, "channels", context)
    channels = {
        channel: _choice(channels_block, channel, CHANNEL_STATES, f"{context}, channels") for channel in CHANNELS
    }
    unknown_channels = set(channels_block) - set(CHANNELS)
    _check(not unknown_channels, f"{context}: nieznane kanały {_listed(sorted(unknown_channels))}")

    manual_events = tuple(
        _manual_event(item, hardware, f"{context}, manual_event nr {number}")
        for number, item in enumerate(block.get("manual_event", []), start=1)
    )
    issues = tuple(
        _issue(item, f"{context}, issue nr {number}") for number, item in enumerate(block.get("issue", []), start=1)
    )

    return Period(
        site=site.id,
        id=period_id,
        legacy_name=_text(block, "legacy_name", context, ""),
        hardware=hardware,
        source=_source(_table(block, "source", context), dumps, context),
        time=time_status,
        clock_offset_h=clock_offset_h,
        time_uncertainty_h=_number(block, "time_uncertainty_h", context),
        start=start,
        end=end,
        checked=_boolean(block, "checked", context),
        note=_text(block, "note", context, ""),
        channels=channels,
        issues=issues,
        manual_events=manual_events,
        catalog_sha256=block_sha256(site.id, block, hardware_blocks[hardware_id]),
    )


def _source(block: dict[str, Any], dumps: dict[str, Dump], context: str) -> Source:
    context = f"{context}, source"
    dump = _text(block, "dump", context)
    _check(dump in dumps, f"{context}: zrzut {dump} nie istnieje w {RAW_FILE}")
    by_session = "session" in block
    by_ids = "first_id" in block or "last_id" in block
    _check(by_session != by_ids, f"{context}: podaj albo session, albo first_id i last_id")

    if by_session:
        return Source(dump=dump, session=_integer(block, "session", context), first_id=None, last_id=None)
    first_id = _integer(block, "first_id", context)
    last_id = _integer(block, "last_id", context)
    _check(first_id <= last_id, f"{context}: first_id większe niż last_id")
    return Source(dump=dump, session=None, first_id=first_id, last_id=last_id)


def _issue(block: dict[str, Any], context: str) -> Issue:
    start = _datetime(block, "start", context)
    end = _datetime(block, "end", context)
    _check(start < end, f"{context}: start nie jest przed end")
    return Issue(
        channel=_choice(block, "channel", CHANNELS, context),
        kind=_text(block, "kind", context),
        start=start,
        end=end,
        note=_text(block, "note", context, ""),
    )


def _manual_event(block: dict[str, Any], hardware: Hardware, context: str) -> ManualEvent:
    channel = _choice(block, "channel", SIGNAL_CHANNELS, context)
    _check(hardware.pin_for(channel) is not None, f"{context}: sprzęt {hardware.id} nie ma pinu dla kanału {channel}")
    start = _datetime(block, "start", context)
    end = _datetime(block, "end", context) if "end" in block else None
    _check(end is None or start < end, f"{context}: start nie jest przed end")
    return ManualEvent(channel=channel, start=start, end=end)


def block_sha256(site_id: str, period_block: dict[str, Any], hardware_block: dict[str, Any]) -> str:
    normalized = {"site": site_id, "period": period_block, "hardware": hardware_block}
    text = json.dumps(normalized, sort_keys=True, ensure_ascii=False, default=_json_value)
    return hashlib.sha256(text.encode()).hexdigest()


def _json_value(value: object) -> str:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"nieobsługiwany typ w katalogu: {type(value).__name__}")


def _read(path: Path) -> dict[str, Any]:
    _check(path.exists(), f"brak pliku katalogu {path}")
    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except tomllib.TOMLDecodeError as error:
        raise CatalogError(f"{path.name}: błąd składni TOML: {error}") from error


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise CatalogError(message)


def _present(block: dict[str, Any], key: str, context: str) -> Any:
    _check(key in block, f"{context}: brak pola {key}")
    return block[key]


def _table(block: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    value = _present(block, key, context)
    _check(isinstance(value, dict), f"{context}: {key} musi być tabelą")
    return value


def _text(block: dict[str, Any], key: str, context: str, default: str | None = None) -> str:
    if default is not None and key not in block:
        return default
    value = _present(block, key, context)
    _check(isinstance(value, str), f"{context}: {key} musi być tekstem")
    return value


def _integer(block: dict[str, Any], key: str, context: str) -> int:
    value = _present(block, key, context)
    _check(isinstance(value, int) and not isinstance(value, bool), f"{context}: {key} musi być liczbą całkowitą")
    return value


def _number(block: dict[str, Any], key: str, context: str) -> float:
    value = _present(block, key, context)
    _check(isinstance(value, int | float) and not isinstance(value, bool), f"{context}: {key} musi być liczbą")
    return float(value)


def _boolean(block: dict[str, Any], key: str, context: str) -> bool:
    value = _present(block, key, context)
    _check(isinstance(value, bool), f"{context}: {key} musi być true albo false")
    return value


def _date(block: dict[str, Any], key: str, context: str) -> date:
    value = _present(block, key, context)
    _check(isinstance(value, date) and not isinstance(value, datetime), f"{context}: {key} musi być datą RRRR-MM-DD")
    return value


def _datetime(block: dict[str, Any], key: str, context: str) -> datetime:
    value = _present(block, key, context)
    _check(isinstance(value, datetime), f"{context}: {key} musi być datą z godziną")
    _check(value.tzinfo is not None, f"{context}: {key} bez strefy czasowej (dopisz np. +02:00)")
    return value


def _choice(block: dict[str, Any], key: str, allowed: tuple[str, ...], context: str) -> str:
    value = _text(block, key, context)
    _check(value in allowed, f"{context}: nieznana wartość {key} = „{value}” (dozwolone: {_listed(allowed)})")
    return value


def _listed(values: tuple[str, ...] | list[str]) -> str:
    return ", ".join(values)
