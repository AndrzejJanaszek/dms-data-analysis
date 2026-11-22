from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from dms_analysis.data import check
from dms_analysis.errors.balance import dispense_masses, inside_refills, overlapping
from dms_analysis.errors.fragments import (
    CLEARANCE_S,
    free_intervals,
    local_moments,
    part_of_hour,
    real_events,
    sample_gaps,
)
from dms_analysis.errors.grid import after_values, before_values
from dms_analysis.events.values import DELAYS_S, DISPENSE, METHODS, REFILL, WINDOW_S, PeriodData
from dms_analysis.raw import TIMEZONE

STITCH_GAP_S = 300.0
STITCH_CHANGE_KG = 100.0

ISSUE = "problem z katalogu"
NEAR_EVENT = "zdarzenie w pobliżu"
SAMPLE_GAP = "luka w próbkach"
FRAGMENT_EDGE = "granica okresu albo części"
CLOCK_CHANGE = "zmiana czasu"


@dataclass(frozen=True)
class LoadedPeriod:
    data: PeriodData
    events: pd.DataFrame
    free: np.ndarray


@dataclass(frozen=True)
class Schedule:
    day: date
    dispenses: pd.DataFrame
    periods: tuple[str, ...]


@dataclass(frozen=True)
class Base:
    day: date
    period: LoadedPeriod


def load(data: PeriodData, marks: pd.DataFrame) -> LoadedPeriod:
    events = real_events(data, marks)
    return LoadedPeriod(data, events, free_intervals(data, events))


def day_bounds(day: date) -> tuple[float, float]:
    start, end = check.day_bounds(day)
    return start.timestamp(), end.timestamp()


def covering_parts(periods: list[LoadedPeriod], day: date) -> list[tuple[float, float, LoadedPeriod]]:
    start, end = day_bounds(day)
    parts = [
        (float(part.start), float(part.end), period)
        for period in periods
        for part in period.data.parts.itertuples()
        if part.start < end and part.end > start
    ]
    return sorted(parts, key=lambda item: item[0])


def is_covered(periods: list[LoadedPeriod], day: date) -> bool:
    start, end = day_bounds(day)
    parts = covering_parts(periods, day)
    if not parts or parts[0][0] > start or parts[-1][1] < end:
        return False

    for (_, left_end, left), (right_start, _, right) in zip(parts, parts[1:], strict=False):
        if right_start - left_end > STITCH_GAP_S:
            return False
        change = right.data.weight.first_from(right_start) - left.data.weight.last_before(left_end + 1e-6)
        if abs(change) > STITCH_CHANGE_KG:
            return False
    return True


def on_day(events: pd.DataFrame, day: date) -> pd.DataFrame:
    start, end = day_bounds(day)
    return events[(events["start"] >= start) & (events["start"] < end)]


def schedules(periods: list[LoadedPeriod], days: list[date]) -> list[Schedule]:
    found = []
    for day in days:
        if not is_covered(periods, day):
            continue
        frames, names = [], []
        for period in periods:
            dispenses = on_day(period.events[period.events["role"] == DISPENSE], day)
            if dispenses.empty:
                continue
            refills = period.events[period.events["role"] == REFILL]
            masses = dispense_masses(period.data, dispenses)
            frames.append(
                dispenses[["start", "end"]].assign(
                    in_refill=inside_refills(dispenses, refills).to_numpy(), mass_kg=masses, period=period.data.full_id
                )
            )
            names.append(period.data.full_id)
        if frames:
            found.append(Schedule(day, pd.concat(frames, ignore_index=True).sort_values("start"), tuple(names)))
    return found


def bases(periods: list[LoadedPeriod], days: list[date]) -> list[Base]:
    found = []
    for day in days:
        start, end = day_bounds(day)
        for period in periods:
            inside = ((period.data.parts["start"] <= start) & (period.data.parts["end"] >= end)).any()
            if inside and overlapping(period.events, start, end).empty:
                found.append(Base(day, period))
    return found


def clock_offsets(starts: np.ndarray, day: date) -> np.ndarray:
    moments = local_moments(starts)
    midnight = pd.Timestamp(day)
    naive = moments.tz_localize(None)
    return (naive - midnight).total_seconds().to_numpy()


def shifted(offsets: np.ndarray, day: date) -> np.ndarray:
    naive = pd.Timestamp(day) + pd.to_timedelta(offsets, unit="s")
    localized = pd.DatetimeIndex(naive).tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
    seconds = localized.as_unit("ns").asi8.astype(float) / 1e9
    return np.where(localized.isna(), np.nan, seconds)


def fits(free: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    if free.size == 0:
        return np.zeros(len(low), dtype=bool)
    index = np.searchsorted(free[:, 0], low, "right") - 1
    valid = index >= 0
    safe = np.clip(index, 0, None)
    return valid & (free[safe, 1] >= high)


def blocking_reason(period: LoadedPeriod, low: float, high: float) -> str:
    if np.isnan(low):
        return CLOCK_CHANGE
    if not overlapping(period.data.issues, low, high).empty:
        return ISSUE
    if not overlapping(period.events, low - CLEARANCE_S, high + CLEARANCE_S).empty:
        return NEAR_EVENT
    gaps = sample_gaps(period.data.weight)
    if ((gaps[:, 0] < high) & (gaps[:, 1] > low)).any():
        return SAMPLE_GAP
    return FRAGMENT_EDGE


@dataclass(frozen=True)
class PairResult:
    valid: bool
    reason: str
    blocked: int
    errors: dict[tuple[str, float], np.ndarray]
    starts: np.ndarray
    durations: np.ndarray


def replay(schedule: Schedule, base: Base) -> PairResult:
    durations = (schedule.dispenses["end"] - schedule.dispenses["start"]).to_numpy(dtype=float)
    starts = shifted(clock_offsets(schedule.dispenses["start"].to_numpy(dtype=float), schedule.day), base.day)
    low = starts - WINDOW_S
    high = starts + durations + max(DELAYS_S) + WINDOW_S

    inside = fits(base.period.free, low, high) & ~np.isnan(starts)
    if not inside.all():
        first = int(np.flatnonzero(~inside)[0])
        reason = blocking_reason(base.period, low[first], high[first])
        return PairResult(False, reason, int((~inside).sum()), {}, starts, durations)

    weight = base.period.data.weight
    errors = {}
    for method in METHODS:
        before, _ = before_values(weight, starts, method)
        for delay_s in DELAYS_S:
            after, _ = after_values(weight, starts + durations, delay_s, method)
            errors[(method, delay_s)] = before - after
    return PairResult(True, "", 0, errors, starts, durations)


def hours_of(starts: np.ndarray) -> np.ndarray:
    return local_moments(starts).hour.to_numpy()


def parts_of(starts: np.ndarray) -> np.ndarray:
    return np.array([part_of_hour(int(hour)) for hour in hours_of(starts)])
