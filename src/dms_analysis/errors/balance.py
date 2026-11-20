from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from dms_analysis.errors.fragments import WEIGHT_CHANNEL, period_dates, review_accepted
from dms_analysis.events.values import (
    BASE_DELAY_S,
    DISPENSE,
    MEDIAN,
    REFILL,
    PeriodData,
    after_value,
    before_value,
)
from dms_analysis.raw import TIMEZONE

NIGHT_TO_NIGHT = "noc–noc"
DAY_ONLY = "5–22"
VARIANTS = (NIGHT_TO_NIGHT, DAY_ONLY)
REFERENCE_WINDOWS = {
    NIGHT_TO_NIGHT: (((0, 0), (4, 0), 0), ((0, 0), (4, 0), 1)),
    DAY_ONLY: (((4, 50), (5, 0), 0), ((22, 0), (22, 10), 0)),
}

CONTROL = "kontrolna"
WITH_REFILL = "z zasypem"
WITHOUT_REFILL = "bez zasypu"
GROUPS = (CONTROL, WITHOUT_REFILL, WITH_REFILL)

OUTSIDE_PERIOD = "poza okresem albo luka w danych"
EVENT_IN_WINDOW = "zdarzenie w oknie odniesienia"
MISSING_MASS = "brak wskazań wokół zdarzenia"

COLUMNS = [
    "variant",
    "date",
    "hours",
    "valid",
    "reason",
    "group",
    "start_window_start",
    "start_window_end",
    "end_window_start",
    "end_window_end",
    "w_start_kg",
    "w_end_kg",
    "change_kg",
    "refills",
    "refill_kg",
    "dispenses",
    "dispense_kg",
    "dispenses_in_refill",
    "review_accepted",
    "disturbance",
    "r_kg",
]


@dataclass(frozen=True)
class Window:
    start: float
    end: float


def local_moment(day: date, hour: int, minute: int) -> float:
    moment = pd.Timestamp(day) + pd.Timedelta(hours=hour, minutes=minute)
    return moment.tz_localize(TIMEZONE).timestamp()


def reference_windows(day: date, variant: str) -> tuple[Window, Window]:
    windows = []
    for (start_h, start_m), (end_h, end_m), offset in REFERENCE_WINDOWS[variant]:
        shifted = day + timedelta(days=offset)
        windows.append(Window(local_moment(shifted, start_h, start_m), local_moment(shifted, end_h, end_m)))
    return windows[0], windows[1]


def day_hours(day: date) -> float:
    return (local_moment(day + timedelta(days=1), 0, 0) - local_moment(day, 0, 0)) / 3600


def overlapping(frame: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    return frame[(frame["start"] < end) & (frame["end"] > start)]


def window_median(data: PeriodData, window: Window) -> float:
    samples = data.weight.window(window.start, window.end)
    return float(np.median(samples)) if samples.size else np.nan


def refill_masses(data: PeriodData, refills: pd.DataFrame) -> np.ndarray:
    return np.array(
        [
            after_value(data.weight, event.end, 0.0, MEDIAN) - before_value(data.weight, event.start, MEDIAN)
            for event in refills.itertuples()
        ],
        dtype=float,
    )


def dispense_masses(data: PeriodData, dispenses: pd.DataFrame) -> np.ndarray:
    return np.array(
        [
            before_value(data.weight, event.start, MEDIAN) - after_value(data.weight, event.end, BASE_DELAY_S, MEDIAN)
            for event in dispenses.itertuples()
        ],
        dtype=float,
    )


def inside_refills(dispenses: pd.DataFrame, refills: pd.DataFrame) -> pd.Series:
    inside = pd.Series(False, index=dispenses.index)
    for refill in refills.itertuples():
        inside |= (dispenses["start"] < refill.end) & (dispenses["end"] > refill.start)
    return inside


def day_balance(data: PeriodData, events: pd.DataFrame, reviewed: pd.DataFrame, day: date, variant: str) -> dict:
    first, last = reference_windows(day, variant)
    row: dict[str, object] = {
        "variant": variant,
        "date": day.isoformat(),
        "hours": day_hours(day),
        "start_window_start": first.start,
        "start_window_end": first.end,
        "end_window_start": last.start,
        "end_window_end": last.end,
    }

    if not _inside_one_part(data, first.start, last.end):
        return _invalid(row, OUTSIDE_PERIOD)
    if not overlapping(events, first.start, first.end).empty or not overlapping(events, last.start, last.end).empty:
        return _invalid(row, EVENT_IN_WINDOW)

    inside = overlapping(events, first.end, last.start)
    refills = inside[inside["role"] == REFILL]
    dispenses = inside[inside["role"] == DISPENSE]
    in_refill = inside_refills(dispenses, refills)
    counted = dispenses[~in_refill]

    refill_kg = refill_masses(data, refills)
    dispense_kg = dispense_masses(data, counted)
    if np.isnan(refill_kg).any() or np.isnan(dispense_kg).any():
        return _invalid(row, MISSING_MASS)

    w_start, w_end = window_median(data, first), window_median(data, last)
    disturbances = data.issues[data.issues["channel"] == WEIGHT_CHANNEL]
    row.update(
        {
            "valid": 1,
            "reason": "",
            "group": _group(refills, dispenses),
            "w_start_kg": w_start,
            "w_end_kg": w_end,
            "change_kg": w_end - w_start,
            "refills": len(refills),
            "refill_kg": float(refill_kg.sum()),
            "dispenses": len(counted),
            "dispense_kg": float(dispense_kg.sum()),
            "dispenses_in_refill": int(in_refill.sum()),
            "review_accepted": len(overlapping(reviewed, first.end, last.start)),
            "disturbance": int(not overlapping(disturbances, first.start, last.end).empty),
            "r_kg": (w_end - w_start) - float(refill_kg.sum()) + float(dispense_kg.sum()),
        }
    )
    return row


def _inside_one_part(data: PeriodData, start: float, end: float) -> bool:
    return bool(((data.parts["start"] <= start) & (data.parts["end"] >= end)).any())


def _invalid(row: dict[str, object], reason: str) -> dict[str, object]:
    return {**row, "valid": 0, "reason": reason}


def _group(refills: pd.DataFrame, dispenses: pd.DataFrame) -> str:
    if not refills.empty:
        return WITH_REFILL
    if not dispenses.empty:
        return WITHOUT_REFILL
    return CONTROL


def period_balance(data: PeriodData, events: pd.DataFrame, marks: pd.DataFrame) -> pd.DataFrame:
    reviewed = review_accepted(data, marks)
    rows = [day_balance(data, events, reviewed, day, variant) for variant in VARIANTS for day in period_dates(data)]
    return pd.DataFrame(rows).reindex(columns=COLUMNS)
