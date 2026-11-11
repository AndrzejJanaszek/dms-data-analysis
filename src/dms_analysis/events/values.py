from dataclasses import dataclass

import numpy as np
import pandas as pd

from dms_analysis.data import check
from dms_analysis.data.catalog import Catalog, Period
from dms_analysis.data.prepare import epoch_seconds

DISPENSE = "odsyp"
REFILL = "zasyp"

WINDOW_S = 3.0
DELAYS_S = (0.0, 2.0, 3.0)
MIN_WINDOW_SAMPLES = 3

NEAREST = "nearest"
MEDIAN = "median"
MEAN = "mean"
METHODS = (NEAREST, MEDIAN, MEAN)
METHOD_NAMES = {NEAREST: "najbliższa próbka", MEDIAN: "mediana 3 s", MEAN: "średnia 3 s"}
REFILL_METHODS = (NEAREST, MEDIAN)

BASE_METHOD = MEDIAN
BASE_DELAY_S = 3.0

EXCLUDED_WINDOW = "okno"
EXCLUDED_COLLISION = "kolizja"
DETECTED_STATES = ("ok", "flickering")

DISPENSE_COLUMNS = ["event", "start", "end", "duration_s", "manual", "excluded", "gap_before_s", "gap_after_s"]
VALUE_COLUMNS = ["event", "method", "delay_s", "before_kg", "after_kg", "mass_kg", "no_drop"]
REFILL_COLUMNS = [
    "event",
    "start",
    "end",
    "duration_s",
    "manual",
    "mass_nearest_kg",
    "mass_median_kg",
    "difference_kg",
    "dispenses_inside",
]


@dataclass(frozen=True)
class Weight:
    seconds: np.ndarray
    values: np.ndarray

    def window(self, start: float, end: float) -> np.ndarray:
        low = np.searchsorted(self.seconds, start, "left")
        high = np.searchsorted(self.seconds, end, "left")
        return self.values[low:high]

    def last_before(self, moment: float) -> float:
        index = np.searchsorted(self.seconds, moment, "left") - 1
        return float(self.values[index]) if index >= 0 else np.nan

    def first_from(self, moment: float) -> float:
        index = np.searchsorted(self.seconds, moment, "left")
        return float(self.values[index]) if index < len(self.values) else np.nan


@dataclass(frozen=True)
class PeriodData:
    full_id: str
    number: int
    time: str
    division_kg: float
    has_temperatures: bool
    weight: Weight
    events: pd.DataFrame
    issues: pd.DataFrame
    parts: pd.DataFrame
    prepared_at: str
    catalog_sha256: str


def stage_periods(catalog: Catalog) -> list[Period]:
    return [
        period
        for period in catalog.periods.values()
        if period.checked and period.channels["dispense"] in DETECTED_STATES
    ]


def load_period_data(period: Period) -> PeriodData:
    prepared = check.load_period(period.full_id)
    weight = prepared.weight.sort_values("timestamp")
    return PeriodData(
        full_id=period.full_id,
        number=period.number,
        time=period.time,
        division_kg=period.hardware.meter_division_kg,
        has_temperatures=period.channels["temperatures"] == "ok",
        weight=Weight(epoch_seconds(weight["timestamp"]).to_numpy(), weight["value"].to_numpy(dtype=float)),
        events=_in_seconds(prepared.events, "start", "end"),
        issues=_in_seconds(prepared.issues, "start", "end"),
        parts=_in_seconds(prepared.parts, "start", "end"),
        prepared_at=str(prepared.info["prepared_at"]),
        catalog_sha256=str(prepared.info["catalog_sha256"]),
    )


def _in_seconds(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    return frame.assign(**{column: epoch_seconds(frame[column]) for column in columns}).reset_index(drop=True)


def accepted(events: pd.DataFrame, role: str | None = None) -> pd.DataFrame:
    chosen = events["rejected"] == 0
    if role is not None:
        chosen &= events["role"] == role
    return events[chosen].sort_values("start").reset_index(drop=True)


def before_value(weight: Weight, start: float, method: str) -> float:
    if method == NEAREST:
        return weight.last_before(start)
    return _statistic(weight.window(start - WINDOW_S, start), method)


def after_value(weight: Weight, end: float, delay_s: float, method: str) -> float:
    if method == NEAREST:
        return weight.first_from(end + delay_s)
    return _statistic(weight.window(end + delay_s, end + delay_s + WINDOW_S), method)


def _statistic(samples: np.ndarray, method: str) -> float:
    if samples.size == 0:
        return np.nan
    if method == MEDIAN:
        return float(np.median(samples))
    return float(np.mean(samples))


def dispense_masses(weight: Weight, start: float, end: float) -> list[dict[str, float | str | bool]]:
    rows = []
    for method in METHODS:
        before = before_value(weight, start, method)
        for delay_s in DELAYS_S:
            after = after_value(weight, end, delay_s, method)
            mass = before - after
            rows.append(
                {
                    "method": method,
                    "delay_s": delay_s,
                    "before_kg": before,
                    "after_kg": after,
                    "mass_kg": mass,
                    "no_drop": bool(mass <= 0),
                }
            )
    return rows


def checked_range(start: float, end: float) -> tuple[float, float]:
    return start - WINDOW_S, end + max(DELAYS_S) + WINDOW_S


def windows(start: float, end: float) -> list[tuple[float, float]]:
    after = [(end + delay_s, end + delay_s + WINDOW_S) for delay_s in DELAYS_S]
    return [(start - WINDOW_S, start), *after]


def exclusion(data: PeriodData, events: pd.DataFrame, position: int) -> str:
    event = events.iloc[position]
    low, high = checked_range(event["start"], event["end"])

    refills = events[events["role"] == REFILL]
    if _overlaps(refills, low, high).any():
        return EXCLUDED_COLLISION

    others = events.drop(index=events.index[position])
    if _overlaps(others, low, high).any() or _overlaps(data.issues, low, high).any():
        return EXCLUDED_WINDOW
    if not _inside_one_part(data.parts, low, high):
        return EXCLUDED_WINDOW
    if any(data.weight.window(a, b).size < MIN_WINDOW_SAMPLES for a, b in windows(event["start"], event["end"])):
        return EXCLUDED_WINDOW
    return ""


def _overlaps(frame: pd.DataFrame, low: float, high: float) -> pd.Series:
    return (frame["start"] < high) & (frame["end"] > low)


def _inside_one_part(parts: pd.DataFrame, low: float, high: float) -> bool:
    return bool(((parts["start"] <= low) & (parts["end"] >= high)).any())


def gaps(events: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    before = events["start"] - events["end"].shift(1)
    after = events["start"].shift(-1) - events["end"]
    return before, after


def dispense_tables(data: PeriodData) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = accepted(data.events)
    gap_before, gap_after = gaps(events)
    table = events.assign(gap_before_s=gap_before, gap_after_s=gap_after)
    dispenses = table[table["role"] == DISPENSE]

    rows, values = [], []
    for number, (position, event) in enumerate(dispenses.iterrows(), start=1):
        rows.append(
            {
                "event": number,
                "start": event["start"],
                "end": event["end"],
                "duration_s": event["end"] - event["start"],
                "manual": int(event["synthetic"]),
                "excluded": exclusion(data, events, int(position)),
                "gap_before_s": event["gap_before_s"],
                "gap_after_s": event["gap_after_s"],
            }
        )
        values.extend({"event": number, **row} for row in dispense_masses(data.weight, event["start"], event["end"]))

    return pd.DataFrame(rows, columns=DISPENSE_COLUMNS), pd.DataFrame(values, columns=VALUE_COLUMNS)


def refill_table(data: PeriodData) -> pd.DataFrame:
    refills = accepted(data.events, REFILL)
    dispenses = accepted(data.events, DISPENSE)
    rows = []
    for number, event in enumerate(refills.itertuples(), start=1):
        masses = {
            method: after_value(data.weight, event.end, 0.0, method) - before_value(data.weight, event.start, method)
            for method in REFILL_METHODS
        }
        rows.append(
            {
                "event": number,
                "start": event.start,
                "end": event.end,
                "duration_s": event.end - event.start,
                "manual": int(event.synthetic),
                "mass_nearest_kg": masses[NEAREST],
                "mass_median_kg": masses[MEDIAN],
                "difference_kg": masses[MEDIAN] - masses[NEAREST],
                "dispenses_inside": int(_overlaps(dispenses, event.start, event.end).sum()),
            }
        )
    return pd.DataFrame(rows, columns=REFILL_COLUMNS)


def base_values(dispenses: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    base = values[(values["method"] == BASE_METHOD) & (values["delay_s"] == BASE_DELAY_S)]
    return dispenses.merge(base, on="event")
