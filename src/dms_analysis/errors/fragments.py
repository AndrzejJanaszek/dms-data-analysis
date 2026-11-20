from datetime import date, timedelta

import numpy as np
import pandas as pd

from dms_analysis.data import check
from dms_analysis.data.review import GOOD, load_marks
from dms_analysis.events.values import DISPENSE, REFILL, PeriodData, Weight, accepted
from dms_analysis.raw import TIMEZONE

DAY_START_H = 5
DAY_END_H = 22
CLEARANCE_S = 60.0
MAX_SAMPLE_STEP_S = 1.5
MARK_TOLERANCE_S = 0.001
FALSE_EVENT = "false_event"
WEIGHT_CHANNEL = "weight"

DAY = "dzień"
NIGHT = "noc"
DAY_PARTS = (DAY, NIGHT)


def real_events(data: PeriodData, marks: pd.DataFrame | None = None) -> pd.DataFrame:
    marks = load_marks() if marks is None else marks
    taken = accepted(data.events)
    taken = taken[~_is_false_event(data, taken)]
    reviewed = review_accepted(data, marks)
    return pd.concat([taken, reviewed]).sort_values("start").reset_index(drop=True)


def review_accepted(data: PeriodData, marks: pd.DataFrame) -> pd.DataFrame:
    rejected = data.events[(data.events["rejected"] == 1) & (data.events["role"] == DISPENSE)]
    good = marks[(marks["period"] == data.full_id) & (marks["mark"] == GOOD)]
    good_starts = (
        pd.to_datetime(good["start"], format="ISO8601", utc=True).map(pd.Timestamp.timestamp).to_numpy(dtype=float)
    )
    if good_starts.size == 0:
        return rejected.iloc[0:0]

    distance = np.abs(rejected["start"].to_numpy(dtype=float)[:, None] - good_starts[None, :])
    return rejected[(distance <= MARK_TOLERANCE_S).any(axis=1)]


def _is_false_event(data: PeriodData, events: pd.DataFrame) -> pd.Series:
    false = data.issues[data.issues["kind"] == FALSE_EVENT]
    inside = pd.Series(False, index=events.index)
    for issue in false.itertuples():
        inside |= (events["start"] < issue.end) & (events["end"] > issue.start)
    return inside


def free_intervals(data: PeriodData, events: pd.DataFrame) -> np.ndarray:
    blocked = [
        np.column_stack([events["start"] - CLEARANCE_S, events["end"] + CLEARANCE_S]),
        data.issues[["start", "end"]].to_numpy(dtype=float).reshape(-1, 2),
        sample_gaps(data.weight),
    ]
    merged = merge(np.concatenate(blocked))

    free = []
    for part in data.parts.itertuples():
        free.extend(subtract(part.start, part.end, merged))
    return np.array(free, dtype=float).reshape(-1, 2)


def sample_gaps(weight: Weight) -> np.ndarray:
    steps = np.diff(weight.seconds)
    gap = np.flatnonzero(steps > MAX_SAMPLE_STEP_S)
    return np.column_stack([weight.seconds[gap], weight.seconds[gap + 1]])


def merge(intervals: np.ndarray) -> np.ndarray:
    if intervals.size == 0:
        return intervals.reshape(0, 2)
    ordered = intervals[np.argsort(intervals[:, 0])]
    merged = [ordered[0].copy()]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append(np.array([start, end]))
    return np.array(merged)


def subtract(start: float, end: float, blocked: np.ndarray) -> list[tuple[float, float]]:
    pieces = []
    cursor = start
    for low, high in blocked:
        if high <= cursor or low >= end:
            continue
        if low > cursor:
            pieces.append((cursor, low))
        cursor = max(cursor, high)
    if cursor < end:
        pieces.append((cursor, end))
    return pieces


def part_of_hour(hour: int) -> str:
    return DAY if DAY_START_H <= hour < DAY_END_H else NIGHT


def local_moments(seconds: np.ndarray) -> pd.DatetimeIndex:
    return pd.to_datetime(seconds, unit="s", utc=True).tz_convert(TIMEZONE)


def period_dates(data: PeriodData) -> list[date]:
    moments = local_moments(data.weight.seconds[[0, -1]])
    first, last = moments[0].date(), moments[1].date()
    return [first + timedelta(days=offset) for offset in range((last - first).days + 1)]


def complete_dates(data: PeriodData) -> set[date]:
    parts = [
        (pd.Timestamp(part.start, unit="s", tz="UTC"), pd.Timestamp(part.end, unit="s", tz="UTC"))
        for part in data.parts.itertuples()
    ]
    return set(check.full_days(parts))


def days_table(data: PeriodData, events: pd.DataFrame, marks: pd.DataFrame) -> pd.DataFrame:
    complete = complete_dates(data)
    reviewed = review_accepted(data, marks)
    disturbances = data.issues[data.issues["channel"] == WEIGHT_CHANNEL]
    rows = []
    for day in period_dates(data):
        start, end = (moment.timestamp() for moment in check.day_bounds(day))
        rows.append(
            {
                "date": day.isoformat(),
                "full_day": int(day in complete),
                "refill": int(_overlapping(events[events["role"] == REFILL], start, end) > 0),
                "disturbance": int(_overlapping(disturbances, start, end) > 0),
                "dispenses": _overlapping(events[events["role"] == DISPENSE], start, end),
                "review_accepted": _overlapping(reviewed, start, end),
            }
        )
    return pd.DataFrame(rows)


def _overlapping(frame: pd.DataFrame, start: float, end: float) -> int:
    return int(((frame["start"] < end) & (frame["end"] > start)).sum())
