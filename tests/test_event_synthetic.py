from datetime import date

import numpy as np
import pandas as pd
import pytest
from event_helpers import events_frame, period_data

from dms_analysis.events import synthetic
from dms_analysis.events.synthetic import DAY, NIGHT, free_intervals, part_segments, period_trials, ramp
from dms_analysis.events.values import DISPENSE, METHODS, Weight

START = pd.Timestamp("2026-11-03 00:00", tz="Europe/Warsaw").timestamp()


def flat_day(value: float = 40000.0) -> Weight:
    seconds = START + np.arange(0, 86400, 1.0)
    return Weight(seconds, np.full(len(seconds), value))


def test_ramp_on_flat_weight_gives_known_mass_exactly() -> None:
    weight = flat_day()

    shifted = ramp(weight, START + 100.0, 30.0, 240.0, 20.0)

    assert shifted.values[99] == 40000.0
    assert shifted.values[131] == 39760.0
    assert set(np.unique(shifted.values % 20)) == {0.0}


def test_trials_on_flat_day_have_only_rounding_error() -> None:
    data = period_data(flat_day(), events_frame([]))
    pool = pd.DataFrame({"mass_kg": [240.0, 480.0], "duration_s": [30.0, 45.0]})

    trials = period_trials(data, pool, trials_per_day_part=20)

    assert len(trials) == 2 * 20 * len(METHODS) * 3
    assert trials["error_kg"].abs().max() <= 10.0 + 1e-9
    assert trials["error_kg"].abs().max() > 0.0
    assert (trials["mass_kg"] % 20 != 0).all()
    assert set(trials["part"]) == {DAY, NIGHT}


def test_trials_are_repeatable_and_keep_clear_of_events() -> None:
    events = events_frame([(DISPENSE, START + 36000.0, START + 36030.0, 0, 0)])
    data = period_data(flat_day(), events)
    pool = pd.DataFrame({"mass_kg": [240.0], "duration_s": [40.0]})

    first = period_trials(data, pool, trials_per_day_part=200)
    second = period_trials(data, pool, trials_per_day_part=200)

    pd.testing.assert_frame_equal(first, second)
    starts = first["start"].to_numpy()
    ends = starts + first["duration_s"].to_numpy() + 6.0
    event_start, event_end = START + 36000.0, START + 36030.0
    clear = (ends <= event_start - synthetic.CLEARANCE_S) | (starts - 3.0 >= event_end + synthetic.CLEARANCE_S)
    assert clear.all()


def test_free_intervals_skip_events_issues_and_sample_gaps() -> None:
    weight = flat_day()
    gapped = Weight(np.delete(weight.seconds, range(5000, 5010)), np.delete(weight.values, range(5000, 5010)))
    events = events_frame([(DISPENSE, START + 1000.0, START + 1030.0, 0, 0)])
    issues = pd.DataFrame(
        {"channel": ["weight"], "kind": ["disturbance"], "start": [START + 3000.0], "end": [START + 3100.0], "note": [""]}
    )

    free = free_intervals(period_data(gapped, events, issues)) - START

    assert free[:4].tolist() == [[0.0, 940.0], [1090.0, 3000.0], [3100.0, 4999.0], [5010.0, 86399.0]]


def test_night_covers_both_ends_of_the_day_on_dst_change() -> None:
    night = part_segments(date(2026, 10, 25), NIGHT)
    day = part_segments(date(2026, 10, 25), DAY)

    assert night[1][1] - night[0][0] == 25 * 3600
    assert day[0][1] - day[0][0] == (synthetic.DAY_END_H - synthetic.DAY_START_H) * 3600
