from pathlib import Path

import pytest

RAW = """
[[dump]]
id = "rpi4/2026-11-09"
device = "rpi4"
pulled = 2026-11-09
file = "rpi4/2026-11-09/measurement-data-saver.db"
sha256 = "fda68914a97108d8ee96773ecc5726584af607de224211b1ec667f9853ba72e6"
program = "measurement-data-saver"
"""

SITE = """
[site]
id = "test-1"
name = "Test"

[[hardware]]
id = "h1"
silo = "silos 1"
meter = "Dini Argeo DGTP"
meter_division_kg = 20
pins = { 5 = "dispense", 4 = "refill" }
temperature_sensors = [{ number = 1, address = "28A8F126AB240B1C", position = "wschód" }]

[[period]]
id = "0001-2026-11-02"
hardware = "h1"
source = { dump = "rpi4/2026-11-09", session = 33 }
time = "real"
clock_offset_h = 0
time_uncertainty_h = 0
start = 2026-11-02T10:42:14+01:00
end = 2026-11-09T13:41:56+01:00
checked = false

[period.channels]
weight = "ok"
dispense = "ok"
refill = "broken"
temperatures = "ok"

[[period.manual_event]]
channel = "refill"
start = 2026-11-02T16:29:01+01:00
"""


@pytest.fixture
def write_catalog(tmp_path: Path):
    def write(site: str = SITE, raw: str = RAW) -> Path:
        (tmp_path / "raw.toml").write_text(raw)
        (tmp_path / "test-1.toml").write_text(site)
        return tmp_path

    return write
