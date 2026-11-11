from datetime import date, timedelta

import numpy as np
import pandas as pd

from dms_analysis.data import check
from dms_analysis.events.values import (
    DELAYS_S,
    WINDOW_S,
    PeriodData,
    Weight,
    accepted,
    base_values,
    dispense_masses,
)
from dms_analysis.raw import TIMEZONE

DAY_START_H = 5
DAY_END_H = 22
TRIALS_PER_DAY_PART = 1000
CLEARANCE_S = 60.0
MAX_SAMPLE_STEP_S = 1.5
SEED = 20261111

DAY = "dzień"
NIGHT = "noc"
DAY_PARTS = (DAY, NIGHT)

TRIAL_COLUMNS = [
    "trial",
    "date",
    "part",
    "full_day",
    "hour",
    "start",
    "duration_s",
    "mass_kg",
    "method",
    "delay_s",
    "estimate_kg",
    "error_kg",
]


def dispense_pool(tables: list[tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    frames = [base_values(dispenses, values) for dispenses, values in tables]
    pooled = pd.concat(frames, ignore_index=True)
    usable = pooled[(pooled["excluded"] == "") & (pooled["mass_kg"] > 0)]
    return usable[["mass_kg", "duration_s"]].reset_index(drop=True)


def free_intervals(data: PeriodData) -> np.ndarray:
    events = accepted(data.events)
    blocked = [
        np.column_stack([events["start"] - CLEARANCE_S, events["end"] + CLEARANCE_S]),
        data.issues[["start", "end"]].to_numpy(dtype=float).reshape(-1, 2),
        _sample_gaps(data.weight),
    ]
    merged = _merge(np.concatenate(blocked))

    free = []
    for part in data.parts.itertuples():
        free.extend(_subtract(part.start, part.end, merged))
    return np.array(free, dtype=float).reshape(-1, 2)


def _sample_gaps(weight: Weight) -> np.ndarray:
    steps = np.diff(weight.seconds)
    gap = np.flatnonzero(steps > MAX_SAMPLE_STEP_S)
    return np.column_stack([weight.seconds[gap], weight.seconds[gap + 1]])


def _merge(intervals: np.ndarray) -> np.ndarray:
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


def _subtract(start: float, end: float, blocked: np.ndarray) -> list[tuple[float, float]]:
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


def part_segments(day: date, part: str) -> list[tuple[float, float]]:
    def moment(hour: int) -> float:
        return (pd.Timestamp(day) + pd.Timedelta(hours=hour)).tz_localize(TIMEZONE).timestamp()

    if part == DAY:
        return [(moment(DAY_START_H), moment(DAY_END_H))]
    return [(moment(0), moment(DAY_START_H)), (moment(DAY_END_H), moment(24))]


def part_of_hour(hour: int) -> str:
    return DAY if DAY_START_H <= hour < DAY_END_H else NIGHT


def period_dates(data: PeriodData) -> list[date]:
    first = pd.Timestamp(data.weight.seconds[0], unit="s", tz="UTC").tz_convert(TIMEZONE).date()
    last = pd.Timestamp(data.weight.seconds[-1], unit="s", tz="UTC").tz_convert(TIMEZONE).date()
    return [first + timedelta(days=offset) for offset in range((last - first).days + 1)]


def complete_dates(data: PeriodData) -> set[date]:
    parts = [
        (pd.Timestamp(part.start, unit="s", tz="UTC"), pd.Timestamp(part.end, unit="s", tz="UTC"))
        for part in data.parts.itertuples()
    ]
    return set(check.full_days(parts))


def ramp(weight: Weight, start: float, duration_s: float, mass_kg: float, division_kg: float) -> Weight:
    progress = np.clip((weight.seconds - start) / duration_s, 0.0, 1.0)
    shifted = weight.values - mass_kg * progress
    return Weight(weight.seconds, np.round(shifted / division_kg) * division_kg)


def neighbourhood(weight: Weight, start: float, end: float) -> Weight:
    low = np.searchsorted(weight.seconds, start - WINDOW_S - 2.0, "left")
    high = np.searchsorted(weight.seconds, end + max(DELAYS_S) + WINDOW_S + 2.0, "right")
    return Weight(weight.seconds[low:high], weight.values[low:high])


def draw_starts(
    free: np.ndarray, segments: list[tuple[float, float]], durations: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    bounds = np.array(segments, dtype=float)
    free_start = np.repeat(free[:, 0], len(bounds))
    free_end = np.repeat(free[:, 1], len(bounds))
    segment_start = np.tile(bounds[:, 0], len(free))
    segment_end = np.tile(bounds[:, 1], len(free))
    low = np.maximum(free_start + WINDOW_S, segment_start)
    tail_s = max(DELAYS_S) + WINDOW_S

    starts = np.full(len(durations), np.nan)
    for index, duration_s in enumerate(durations):
        high = np.minimum(free_end - duration_s - tail_s, segment_end)
        lengths = np.clip(high - low, 0.0, None)
        total = lengths.sum()
        if total <= 0:
            continue
        chosen = rng.choice(len(lengths), p=lengths / total)
        starts[index] = low[chosen] + rng.uniform() * lengths[chosen]
    return starts


def mass_jitter(division_kg: float, count: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-division_kg / 2, division_kg / 2, count)


def period_trials(data: PeriodData, pool: pd.DataFrame, trials_per_day_part: int = TRIALS_PER_DAY_PART) -> pd.DataFrame:
    free = free_intervals(data)
    complete = complete_dates(data)
    rows = []
    for day in period_dates(data):
        for part_index, part in enumerate(DAY_PARTS):
            rng = np.random.default_rng([SEED, data.number, day.toordinal(), part_index])
            picks = pool.iloc[rng.integers(0, len(pool), trials_per_day_part)]
            durations = picks["duration_s"].to_numpy(dtype=float)
            masses = picks["mass_kg"].to_numpy(dtype=float) + mass_jitter(data.division_kg, len(picks), rng)
            starts = draw_starts(free, part_segments(day, part), durations, rng)
            for trial, (start, duration_s, mass_kg) in enumerate(zip(starts, durations, masses, strict=True), 1):
                if np.isnan(start):
                    continue
                rows.extend(_trial_rows(data, day, part, day in complete, trial, start, duration_s, mass_kg))
    return pd.DataFrame(rows, columns=TRIAL_COLUMNS)


def _trial_rows(
    data: PeriodData,
    day: date,
    part: str,
    full_day: bool,
    trial: int,
    start: float,
    duration_s: float,
    mass_kg: float,
) -> list[dict[str, object]]:
    end = start + duration_s
    synthetic = ramp(neighbourhood(data.weight, start, end), start, duration_s, mass_kg, data.division_kg)
    hour = pd.Timestamp(start, unit="s", tz="UTC").tz_convert(TIMEZONE).hour
    common = {
        "trial": trial,
        "date": day.isoformat(),
        "part": part,
        "full_day": int(full_day),
        "hour": hour,
        "start": start,
        "duration_s": duration_s,
        "mass_kg": mass_kg,
    }
    return [
        {
            **common,
            "method": value["method"],
            "delay_s": value["delay_s"],
            "estimate_kg": value["mass_kg"],
            "error_kg": float(value["mass_kg"]) - mass_kg,
        }
        for value in dispense_masses(synthetic, start, end)
    ]
