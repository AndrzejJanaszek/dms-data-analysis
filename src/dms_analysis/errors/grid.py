from dataclasses import dataclass

import numpy as np
import pandas as pd

from dms_analysis.errors.fragments import DAY_PARTS, local_moments, part_of_hour
from dms_analysis.events.values import (
    BASE_DELAY_S,
    BASE_METHOD,
    DELAYS_S,
    MEDIAN,
    METHODS,
    MIN_WINDOW_SAMPLES,
    NEAREST,
    WINDOW_S,
    Weight,
)

DURATIONS_S = (10, 20, 30, 45, 60, 90)
WORST_PER_GROUP = 20
WORST_SEPARATION_S = 120.0
WORST_CANDIDATES = 20000
ERROR_DECIMALS = 6

COUNT_COLUMNS = ["date", "hour", "duration_s", "delay_s", "method", "error_kg", "starts"]
SKIPPED_COLUMNS = ["date", "hour", "duration_s", "starts"]
WORST_COLUMNS = ["date", "hour", "part", "duration_s", "start", "error_kg"]


@dataclass(frozen=True)
class GridResult:
    counts: pd.DataFrame
    skipped: pd.DataFrame
    worst: pd.DataFrame


def grid_starts(free: np.ndarray, duration_s: float) -> np.ndarray:
    tail_s = duration_s + max(DELAYS_S) + WINDOW_S
    pieces = [
        np.arange(np.ceil(start + WINDOW_S), np.floor(end - tail_s) + 1.0)
        for start, end in free
        if np.floor(end - tail_s) >= np.ceil(start + WINDOW_S)
    ]
    return np.concatenate(pieces) if pieces else np.empty(0)


def window_statistic(weight: Weight, low_s: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    low = np.searchsorted(weight.seconds, low_s, "left")
    high = np.searchsorted(weight.seconds, low_s + WINDOW_S, "left")
    counts = high - low

    result = np.full(len(low_s), np.nan)
    for size in np.unique(counts[counts > 0]):
        chosen = counts == size
        samples = weight.values[low[chosen, None] + np.arange(size)]
        result[chosen] = np.median(samples, axis=1) if method == MEDIAN else np.mean(samples, axis=1)
    return result, counts


def before_values(weight: Weight, starts: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    if method != NEAREST:
        return window_statistic(weight, starts - WINDOW_S, method)
    index = np.searchsorted(weight.seconds, starts, "left") - 1
    return weight.values[index], np.full(len(starts), MIN_WINDOW_SAMPLES)


def after_values(weight: Weight, ends: np.ndarray, delay_s: float, method: str) -> tuple[np.ndarray, np.ndarray]:
    if method != NEAREST:
        return window_statistic(weight, ends + delay_s, method)
    index = np.searchsorted(weight.seconds, ends + delay_s, "left")
    return weight.values[index], np.full(len(ends), MIN_WINDOW_SAMPLES)


def start_errors(
    weight: Weight, starts: np.ndarray, duration_s: float
) -> tuple[dict[tuple[str, float], np.ndarray], np.ndarray]:
    ends = starts + duration_s
    _, before_counts = window_statistic(weight, starts - WINDOW_S, MEDIAN)
    usable = before_counts >= MIN_WINDOW_SAMPLES
    for delay_s in DELAYS_S:
        _, after_counts = window_statistic(weight, ends + delay_s, MEDIAN)
        usable &= after_counts >= MIN_WINDOW_SAMPLES

    errors = {}
    for method in METHODS:
        before, _ = before_values(weight, starts[usable], method)
        for delay_s in DELAYS_S:
            after, _ = after_values(weight, ends[usable], delay_s, method)
            errors[(method, delay_s)] = np.round(before - after, ERROR_DECIMALS)
    return errors, usable


def period_grid(weight: Weight, free: np.ndarray) -> GridResult:
    counts, skipped, worst = [], [], []
    for duration_s in DURATIONS_S:
        starts = grid_starts(free, duration_s)
        errors, usable = start_errors(weight, starts, duration_s)
        labels = _labels(starts)

        skipped.append(_count(labels[~usable].assign(duration_s=duration_s), ["date", "hour", "duration_s"]))
        used = labels[usable].reset_index(drop=True)
        for (method, delay_s), error in errors.items():
            frame = used.assign(duration_s=duration_s, delay_s=delay_s, method=method, error_kg=error)
            counts.append(_count(frame, ["date", "hour", "duration_s", "delay_s", "method", "error_kg"]))
        base = used.assign(duration_s=duration_s, start=starts[usable], error_kg=errors[(BASE_METHOD, BASE_DELAY_S)])
        worst.append(_worst(base))

    return GridResult(
        counts=_concat(counts, COUNT_COLUMNS),
        skipped=_concat(skipped, SKIPPED_COLUMNS),
        worst=_concat(worst, WORST_COLUMNS),
    )


def _labels(starts: np.ndarray) -> pd.DataFrame:
    moments = local_moments(starts)
    hours = moments.hour.to_numpy()
    return pd.DataFrame({"date": moments.strftime("%Y-%m-%d"), "hour": hours})


def _count(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return frame.groupby(keys, sort=True).size().rename("starts").reset_index()


def _worst(base: pd.DataFrame) -> pd.DataFrame:
    base = base.assign(part=base["hour"].map(part_of_hour), size=base["error_kg"].abs())
    chosen = [_separated(base[base["part"] == part]) for part in DAY_PARTS]
    return pd.concat(chosen)[WORST_COLUMNS]


def _separated(candidates: pd.DataFrame) -> pd.DataFrame:
    ordered = candidates.nlargest(WORST_CANDIDATES, "size")
    kept: list[int] = []
    for position, start in enumerate(ordered["start"].to_numpy()):
        if all(abs(start - ordered["start"].iloc[other]) >= WORST_SEPARATION_S for other in kept):
            kept.append(position)
        if len(kept) == WORST_PER_GROUP:
            break
    return ordered.iloc[kept]


def _concat(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True)[columns] if frames else pd.DataFrame(columns=columns)
