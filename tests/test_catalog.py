import pytest
from conftest import SITE

from dms_analysis.data.catalog import CatalogError, load_catalog


def test_valid_catalog_loads(write_catalog):
    catalog = load_catalog(write_catalog())
    period = catalog.period("test-1/0001-2026-11-02")

    assert period.hardware.pins == {5: "dispense", 4: "refill"}
    assert period.hardware.pin_for("refill") == 4
    assert period.manual_events[0].end is None
    assert period.source.session == 33


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('time = "real"\n', "", "brak pola time"),
        ('dispense = "ok"', 'dispense = "works"', "nieznana wartość dispense"),
        ('source = { dump = "rpi4/2026-11-09"', 'source = { dump = "rpi4/2099-01-01"', "nie istnieje w raw.toml"),
        ('hardware = "h1"\nsource', 'hardware = "h2"\nsource', "nieistniejący sprzęt h2"),
        ("pins = { 5 =", 'pins = { "x" =', "nie jest liczbą"),
        ('4 = "refill"', '4 = "valve"', "nieznana rola pinu 4"),
        ('time = "real"', 'time = "unknown"', None),
        ("start = 2026-11-02T10:42:14+01:00", "start = 2026-11-02T10:42:14", "bez strefy czasowej"),
    ],
)
def test_invalid_catalog_is_rejected(write_catalog, old, new, message):
    assert old in SITE
    site = SITE.replace(old, new, 1)

    if message is None:
        load_catalog(write_catalog(site))
        return
    with pytest.raises(CatalogError, match=message):
        load_catalog(write_catalog(site))


def test_unknown_time_requires_zero_offset(write_catalog):
    site = SITE.replace('time = "real"', 'time = "unknown"').replace("clock_offset_h = 0", "clock_offset_h = 2.5")

    with pytest.raises(CatalogError, match="clock_offset_h musi być 0"):
        load_catalog(write_catalog(site))


def test_repeated_period_id_is_rejected(write_catalog):
    period = SITE[SITE.index("[[period]]") :]

    with pytest.raises(CatalogError, match="powtórzony identyfikator okresu"):
        load_catalog(write_catalog(SITE + "\n" + period))


def test_hash_ignores_formatting_but_follows_hardware(write_catalog):
    original = load_catalog(write_catalog()).period("test-1/0001-2026-11-02").catalog_sha256

    reformatted = SITE.replace('time = "real"', 'time   =   "real"   # komentarz')
    assert load_catalog(write_catalog(reformatted)).period("test-1/0001-2026-11-02").catalog_sha256 == original

    other_sensor = SITE.replace('position = "wschód"', 'position = "zachód"')
    assert load_catalog(write_catalog(other_sensor)).period("test-1/0001-2026-11-02").catalog_sha256 != original
