import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from dms_analysis.data.catalog import Hardware

WINDOWS_S = (30, 60)
MAX_FILLED_GAP_S = 10
PLANE_SIGNS = {
    "east_west": {"NE": 1, "SE": 1, "NW": -1, "SW": -1},
    "north_south": {"NW": 1, "NE": 1, "SW": -1, "SE": -1},
    "twist": {"NW": 1, "SE": 1, "NE": -1, "SW": -1},
}


def features(temperatures: pd.DataFrame, hardware: Hardware) -> pd.DataFrame:
    numbers = [sensor.number for sensor in hardware.temperature_sensors]
    values, filled = sensor_grid(temperatures, numbers)
    corners = {sensor.corner: sensor.number for sensor in hardware.temperature_sensors if sensor.corner is not None}

    columns: dict[str, pd.Series] = {}
    for number in numbers:
        columns[f"t{number}"] = values[number]
        columns[f"filled_{number}"] = filled[number].astype(int)
    columns |= direct(values, corners, prefix="")

    for window in WINDOWS_S:
        rates = pd.DataFrame({number: rate(values[number], window) for number in numbers})
        for number in numbers:
            columns[f"rate{window}_{number}"] = rates[number]
        columns |= summary(rates, corners, prefix=f"rate{window}_")

    frame = pd.DataFrame(columns)
    return frame.rename_axis("timestamp").reset_index()


def sensor_grid(temperatures: pd.DataFrame, numbers: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    second = temperatures["timestamp"].round()
    nearest = (
        temperatures.assign(second=second, distance=(temperatures["timestamp"] - second).abs())
        .sort_values("distance")
        .drop_duplicates(["sensor", "second"])
    )

    grid = np.arange(second.min(), second.max() + 1)
    measured = nearest.pivot(index="second", columns="sensor", values="value").reindex(index=grid, columns=numbers)

    values = measured.apply(fill_short_gaps)
    filled = measured.isna() & values.notna()
    return values, filled


def fill_short_gaps(series: pd.Series) -> pd.Series:
    known = series.notna()
    run = (known != known.shift()).cumsum()
    run_length = series.groupby(run).transform("size")
    inside = known.cummax() & known[::-1].cummax()[::-1]
    short = ~known & inside & (run_length < MAX_FILLED_GAP_S)

    interpolated = series.interpolate(limit_area="inside")
    return series.where(~short, interpolated)


def direct(values: pd.DataFrame, corners: dict[str, int], prefix: str) -> dict[str, pd.Series]:
    complete = values.notna().all(axis=1)
    mean = values.mean(axis=1).where(complete)

    columns = {f"{prefix}mean": mean}
    for number in values.columns:
        columns[f"{prefix}dev_{number}"] = values[number] - mean
    columns[f"{prefix}spread"] = values.std(axis=1, ddof=0).where(complete)
    columns[f"{prefix}max"] = values.max(axis=1).where(complete)
    columns[f"{prefix}min"] = values.min(axis=1).where(complete)
    columns[f"{prefix}range"] = columns[f"{prefix}max"] - columns[f"{prefix}min"]
    return columns | plane(values, corners, prefix)


def summary(rates: pd.DataFrame, corners: dict[str, int], prefix: str) -> dict[str, pd.Series]:
    complete = rates.notna().all(axis=1)
    columns = {
        f"{prefix}mean": rates.mean(axis=1).where(complete),
        f"{prefix}spread": rates.std(axis=1, ddof=0).where(complete),
    }
    return columns | plane(rates, corners, prefix)


def plane(values: pd.DataFrame, corners: dict[str, int], prefix: str) -> dict[str, pd.Series]:
    if set(corners) != {"NW", "NE", "SW", "SE"} or len(values.columns) != 4:
        return {f"{prefix}{name}": pd.Series(np.nan, index=values.index) for name in PLANE_SIGNS}

    return {
        f"{prefix}{name}": sum(sign * values[corners[corner]] for corner, sign in signs.items()) / 2
        for name, signs in PLANE_SIGNS.items()
    }


def rate(series: pd.Series, window_s: int) -> pd.Series:
    offsets = np.arange(window_s + 1) - window_s / 2
    weights = offsets / (offsets**2).sum()

    result = np.full(len(series), np.nan)
    if len(series) > window_s:
        windows = sliding_window_view(series.to_numpy(dtype=float), window_s + 1)
        half = window_s // 2
        result[half : half + len(windows)] = windows @ weights * 60

    return pd.Series(result, index=series.index)
