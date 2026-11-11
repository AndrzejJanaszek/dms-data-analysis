import sqlite3
from datetime import date

import pandas as pd
import pytest
from conftest import SITE

from dms_analysis.data import check
from dms_analysis.data.catalog import load_catalog
from dms_analysis.data.check import full_days
from dms_analysis.data.prepare import (
    load_raw,
    period_row,
    stored_catalog_sha256,
    time_label,
    write_database,
)

CLOCK_START = 1_790_000_000.0


@pytest.fixture
def raw_database(tmp_path):
    path = tmp_path / "period.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE measurements (id INTEGER PRIMARY KEY, timestamp REAL, value REAL)")
    connection.executemany(
        "INSERT INTO measurements (timestamp, value) VALUES (?, ?)",
        [(CLOCK_START + second, 100.0) for second in range(3)],
    )
    connection.commit()
    connection.close()
    return path


@pytest.mark.parametrize("offset_h", [0.0, 115.25])
def test_clock_offset_is_added(raw_database, offset_h):
    raw = load_raw(raw_database, offset_h)

    expected = pd.Timestamp(CLOCK_START + offset_h * 3600, unit="s", tz="UTC")
    assert raw.weight["timestamp"].iloc[0] == expected
    assert raw.pins.empty and raw.temperatures.empty


@pytest.mark.parametrize(
    ("status", "offset", "uncertainty", "label"),
    [
        ("real", "0", "0", "czas rzeczywisty"),
        ("estimated", "115.25", "2.5", "czas szacowany ±2,5 h"),
        ("unknown", "0", "0", "czas zegara"),
    ],
)
def test_time_label(write_catalog, status, offset, uncertainty, label):
    site = (
        SITE.replace('time = "real"', f'time = "{status}"')
        .replace("clock_offset_h = 0", f"clock_offset_h = {offset}")
        .replace("time_uncertainty_h = 0", f"time_uncertainty_h = {uncertainty}")
    )
    period = load_catalog(write_catalog(site)).period("test-1/0001-2026-11-02")

    assert time_label(period) == label


def test_catalog_change_is_detected(write_catalog, tmp_path):
    prepared = tmp_path / "prepared.db"
    period = load_catalog(write_catalog()).period("test-1/0001-2026-11-02")
    write_database(prepared, {"period": period_row(period)})

    assert stored_catalog_sha256(prepared) == period.catalog_sha256

    changed = load_catalog(write_catalog(SITE.replace("checked = false", "checked = true")))
    assert stored_catalog_sha256(prepared) != changed.period("test-1/0001-2026-11-02").catalog_sha256


def test_full_days_across_daylight_saving_change():
    start = pd.Timestamp("2026-10-23 12:00", tz="Europe/Warsaw")
    end = pd.Timestamp("2026-10-27 00:00", tz="Europe/Warsaw")

    assert full_days([(start, end)]) == [date(2026, 10, 24), date(2026, 10, 25), date(2026, 10, 26)]


def test_day_split_by_gap_is_not_full():
    first = (pd.Timestamp("2026-11-02 08:00", tz="Europe/Warsaw"), pd.Timestamp("2026-11-03 12:00", tz="Europe/Warsaw"))
    second = (
        pd.Timestamp("2026-11-03 12:05", tz="Europe/Warsaw"),
        pd.Timestamp("2026-11-05 01:00", tz="Europe/Warsaw"),
    )

    assert full_days([first, second]) == [date(2026, 11, 4)]


def test_full_day_with_issue_is_marked():
    zone = "Europe/Warsaw"
    weight = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2026-10-04 12:00", "2026-10-07 12:00"]).tz_localize(zone), "value": [1.0, 1.0]}
    )
    issues = pd.DataFrame(
        {
            "channel": ["weight"],
            "kind": ["disturbance"],
            "start": [pd.Timestamp("2026-10-05 05:48", tz=zone)],
            "end": [pd.Timestamp("2026-10-05 06:11", tz=zone)],
        }
    )
    parts = pd.DataFrame(
        {
            "part": [1],
            "start": weight["timestamp"].iloc[[0]].to_numpy(),
            "end": weight["timestamp"].iloc[[1]].to_numpy(),
        }
    )
    parts = parts.assign(
        start=pd.to_datetime(parts["start"], utc=True).dt.tz_convert(zone),
        end=pd.to_datetime(parts["end"], utc=True).dt.tz_convert(zone),
    )
    empty = pd.DataFrame()
    period = check.PreparedPeriod(
        info=pd.Series({"time": "real", "label": "test"}),
        weight=weight,
        temperatures=empty,
        removed=empty,
        changes=empty,
        events=empty,
        issues=issues,
        parts=parts,
    )

    statuses = dict(zip(check.days_table(period)["doba"].str[:10], check.days_table(period)["status"], strict=True))

    assert statuses == {
        "2026-10-04": check.PARTIAL_DAY,
        "2026-10-05": check.FULL_DAY_WITH_ISSUE,
        "2026-10-06": check.FULL_DAY,
        "2026-10-07": check.PARTIAL_DAY,
    }
