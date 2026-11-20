from datetime import date

import numpy as np
import pandas as pd
from event_helpers import events_frame, period_data

from dms_analysis.errors.balance import (
    CONTROL,
    DAY_ONLY,
    EVENT_IN_WINDOW,
    NIGHT_TO_NIGHT,
    OUTSIDE_PERIOD,
    WITH_REFILL,
    day_balance,
    local_moment,
)
from dms_analysis.events.values import DISPENSE, REFILL, Weight

DAY_ONE = date(2026, 11, 3)
DAY_TWO = date(2026, 11, 4)
NO_MARKS = events_frame([])


def at(day: date, hour: int, minute: int = 0) -> float:
    return local_moment(day, hour, minute)


def stepped_weight(steps: list[tuple[float, float, float]]) -> Weight:
    seconds = np.arange(at(DAY_ONE, 0), at(date(2026, 11, 5), 6), 1.0)
    values = np.full(len(seconds), 40000.0)
    for start, end, change in steps:
        progress = np.clip((seconds - start) / max(end - start, 1.0), 0.0, 1.0)
        values += change * progress
    return Weight(seconds, values)


def scenario() -> tuple[Weight, pd.DataFrame]:
    dispense = (at(DAY_ONE, 10), at(DAY_ONE, 10) + 30.0)
    refill = (at(DAY_ONE, 13), at(DAY_ONE, 13, 20))
    steps = [
        (at(DAY_ONE, 8), at(DAY_ONE, 8) + 600.0, 40.0),
        (dispense[0], dispense[1], -500.0),
        (refill[0], refill[1], 3000.0),
        (at(DAY_ONE, 16), at(DAY_ONE, 16) + 900.0, -60.0),
        (at(DAY_TWO, 12), at(DAY_TWO, 12) + 600.0, 20.0),
    ]
    events = events_frame([(DISPENSE, *dispense, 0, 0), (REFILL, *refill, 0, 1)])
    return stepped_weight(steps), events


def test_balance_equals_sum_of_changes_between_events() -> None:
    weight, events = scenario()
    data = period_data(weight, events)

    night = day_balance(data, events, NO_MARKS, DAY_ONE, NIGHT_TO_NIGHT)
    day = day_balance(data, events, NO_MARKS, DAY_ONE, DAY_ONLY)

    assert night["valid"] == 1 and night["group"] == WITH_REFILL
    assert night["r_kg"] == -20.0
    assert night["refill_kg"] == 3000.0 and night["dispense_kg"] == 500.0
    assert day["r_kg"] == -20.0


def test_control_day_is_change_between_nights() -> None:
    weight, events = scenario()
    data = period_data(weight, events)

    control = day_balance(data, events, NO_MARKS, DAY_TWO, NIGHT_TO_NIGHT)

    assert control["group"] == CONTROL
    assert control["r_kg"] == control["w_end_kg"] - control["w_start_kg"] == 20.0


def test_day_without_next_night_or_with_event_in_window_is_invalid() -> None:
    weight, events = scenario()
    late = events_frame([(DISPENSE, at(DAY_TWO, 2), at(DAY_TWO, 2) + 30.0, 0, 0)])
    data = period_data(weight, pd.concat([events, late], ignore_index=True))

    outside = day_balance(data, data.events, NO_MARKS, date(2026, 11, 5), NIGHT_TO_NIGHT)
    in_window = day_balance(data, data.events, NO_MARKS, DAY_ONE, NIGHT_TO_NIGHT)

    assert outside["valid"] == 0 and outside["reason"] == OUTSIDE_PERIOD
    assert in_window["valid"] == 0 and in_window["reason"] == EVENT_IN_WINDOW


def test_dispense_inside_refill_is_part_of_refill_block() -> None:
    weight, events = scenario()
    inside = events_frame([(DISPENSE, at(DAY_ONE, 13, 5), at(DAY_ONE, 13, 5) + 30.0, 0, 0)])
    data = period_data(weight, pd.concat([events, inside], ignore_index=True))

    night = day_balance(data, data.events, NO_MARKS, DAY_ONE, NIGHT_TO_NIGHT)

    assert night["dispenses"] == 1
    assert night["dispenses_in_refill"] == 1
    assert night["r_kg"] == -20.0
