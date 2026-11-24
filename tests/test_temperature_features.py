import numpy as np
import pandas as pd
import pytest

from dms_analysis.data.catalog import Hardware, TemperatureSensor
from dms_analysis.temperatures.features import features, rate, sensor_grid

CORNERS = {1: "SE", 2: "SW", 3: "NW", 4: "NE"}


def hardware(corners: dict[int, str | None]) -> Hardware:
    sensors = tuple(
        TemperatureSensor(number=number, address=f"28{number:014d}", position="", corner=corner)
        for number, corner in corners.items()
    )
    return Hardware(id="h1", silo="", meter="", meter_division_kg=20, pins={}, temperature_sensors=sensors)


def readings(values: dict[int, np.ndarray], jitter: float = 0.0) -> pd.DataFrame:
    frames = [
        pd.DataFrame({"timestamp": np.arange(len(series)) + 1000 + jitter, "sensor": number, "value": series})
        for number, series in values.items()
    ]
    return pd.concat(frames, ignore_index=True).dropna()


def test_rate_of_a_ramp_is_its_slope_in_degrees_per_minute():
    ramp = pd.Series(np.arange(200) * 0.01)

    result = rate(ramp, 30)

    assert result.iloc[:15].isna().all()
    assert result.iloc[-15:].isna().all()
    assert result.iloc[15:-15].to_numpy() == pytest.approx(0.6)


def test_rate_is_empty_when_window_touches_a_gap():
    series = pd.Series(np.ones(200))
    series.iloc[100] = np.nan

    result = rate(series, 30)

    assert result.iloc[85:116].isna().all()
    assert result.iloc[84] == pytest.approx(0)
    assert result.iloc[116] == pytest.approx(0)


def test_gaps_up_to_ten_seconds_are_interpolated_and_flagged():
    series = np.arange(100, dtype=float)
    series[20:29] = np.nan
    series[50:60] = np.nan

    values, filled = sensor_grid(readings({1: series}, jitter=0.3), [1])

    assert values[1].iloc[20:29].to_numpy() == pytest.approx(np.arange(20, 29))
    assert filled[1].iloc[20:29].all()
    assert values[1].iloc[50:60].isna().all()
    assert not filled[1].iloc[50:60].any()
    assert not filled[1].iloc[:20].any()


def test_nearest_reading_wins_when_two_fall_on_one_second():
    temperatures = pd.DataFrame({"timestamp": [1000.0, 1000.4, 1001.1], "sensor": 1, "value": [5.0, 7.0, 6.0]})

    values, _ = sensor_grid(temperatures, [1])

    assert values[1].tolist() == [5.0, 6.0]


def test_spread_squared_equals_quarter_of_plane_components_squared():
    rng = np.random.default_rng(1)
    values = {number: rng.normal(10, 2, 300) for number in CORNERS}

    frame = features(readings(values), hardware(CORNERS))

    components = frame[["east_west", "north_south", "twist"]].pow(2).sum(axis=1) / 4
    assert frame["spread"].pow(2).to_numpy() == pytest.approx(components.to_numpy())
    expected_east_west = ((frame["t4"] + frame["t1"]) - (frame["t3"] + frame["t2"])) / 2
    assert frame["east_west"].to_numpy() == pytest.approx(expected_east_west.to_numpy())
    rates = frame[[f"rate60_{number}" for number in CORNERS]]
    assert frame["rate60_mean"].to_numpy() == pytest.approx(rates.mean(axis=1).to_numpy(), nan_ok=True)


def test_missing_sensor_empties_combined_values_but_keeps_its_neighbours():
    values = {number: np.full(200, 10.0 + number) for number in CORNERS}
    values[2][50:80] = np.nan

    frame = features(readings(values), hardware(CORNERS))

    assert frame["mean"].iloc[50:80].isna().all()
    assert frame["east_west"].iloc[50:80].isna().all()
    assert frame["mean"].iloc[10] == pytest.approx(12.5)
    assert frame["t1"].iloc[50:80].notna().all()
    assert frame["rate30_1"].iloc[50:80].notna().all()


def test_plane_is_empty_without_corners():
    values = {number: np.full(100, 10.0) for number in CORNERS}

    frame = features(readings(values), hardware({number: None for number in CORNERS}))

    assert frame["north_south"].isna().all()
    assert frame["spread"].notna().all()
