from datetime import date

import numpy as np
import pandas as pd
from event_helpers import events_frame, period_data

from dms_analysis.errors import stats
from dms_analysis.errors.balance import local_moment
from dms_analysis.errors.schedule import (
    FRAGMENT_EDGE,
    ISSUE,
    SAMPLE_GAP,
    Base,
    Schedule,
    load,
    replay,
    schedules,
)
from dms_analysis.events.values import DISPENSE, METHODS, Weight, dispense_masses

WORK_DAY = date(2026, 11, 3)
QUIET_DAY = date(2026, 11, 8)
NO_MARKS = pd.DataFrame(columns=["period", "start", "end", "mark"])


def noisy_day(day: date, seed: int, drop: list[tuple[float, float]] | None = None) -> Weight:
    rng = np.random.default_rng(seed)
    seconds = np.arange(local_moment(day, 0, 0) - 3600.0, local_moment(day, 23, 59) + 3600.0, 1.0)
    seconds = seconds + rng.uniform(0.0, 0.002, len(seconds))
    values = np.round((40000.0 + np.cumsum(rng.normal(0.0, 5.0, len(seconds)))) / 20.0) * 20.0
    keep = np.ones(len(seconds), dtype=bool)
    for start, end in drop or []:
        keep &= ~((seconds >= start) & (seconds < end))
    return Weight(seconds[keep], values[keep])


def schedule_of(starts: list[tuple[int, int]], duration_s: float = 30.0) -> Schedule:
    moments = np.array([local_moment(WORK_DAY, hour, minute) for hour, minute in starts])
    dispenses = pd.DataFrame(
        {"start": moments, "end": moments + duration_s, "in_refill": False, "mass_kg": 500.0, "period": "p"}
    )
    return Schedule(WORK_DAY, dispenses, ("p",))


def quiet_base(issues: pd.DataFrame | None = None, drop: list[tuple[float, float]] | None = None) -> Base:
    data = period_data(noisy_day(QUIET_DAY, 7, drop), events_frame([]), issues)
    return Base(QUIET_DAY, load(data, NO_MARKS))


def test_pair_error_is_sum_of_single_dispense_errors() -> None:
    schedule = schedule_of([(8, 0), (9, 30), (14, 15)])
    base = quiet_base()

    result = replay(schedule, base)

    assert result.valid
    expected_starts = [local_moment(QUIET_DAY, hour, minute) for hour, minute in ((8, 0), (9, 30), (14, 15))]
    assert np.allclose(result.starts, expected_starts)
    for method in METHODS:
        for start, error in zip(result.starts, result.errors[(method, 3.0)], strict=True):
            rows = dispense_masses(base.period.data.weight, start, start + 30.0)
            single = next(row for row in rows if row["method"] == method and row["delay_s"] == 3.0)
            assert np.isclose(error, single["mass_kg"])


def test_pair_is_invalid_on_issue_gap_and_period_edge() -> None:
    schedule = schedule_of([(10, 0)])
    moment = local_moment(QUIET_DAY, 10, 0)
    issue = pd.DataFrame(
        [("weight", "disturbance", moment - 5.0, moment + 5.0, "")], columns=["channel", "kind", "start", "end", "note"]
    )

    on_issue = replay(schedule, quiet_base(issues=issue))
    on_gap = replay(schedule, quiet_base(drop=[(moment + 10.0, moment + 13.0)]))
    late = schedule_of([(23, 0)], duration_s=90.0)
    end = local_moment(QUIET_DAY, 23, 1)
    on_edge = replay(late, quiet_base(drop=[(end, end + 86400.0)]))

    assert (on_issue.valid, on_issue.reason) == (False, ISSUE)
    assert (on_gap.valid, on_gap.reason) == (False, SAMPLE_GAP)
    assert (on_edge.valid, on_edge.reason) == (False, FRAGMENT_EDGE)


def test_schedules_take_real_dispenses_of_full_days() -> None:
    start = local_moment(WORK_DAY, 9, 0)
    events = events_frame([(DISPENSE, start, start + 30.0, 0, 0), (DISPENSE, start + 600.0, start + 603.0, 1, 0)])
    period = load(period_data(noisy_day(WORK_DAY, 3), events), NO_MARKS)

    found = schedules([period], [WORK_DAY])

    assert len(found) == 1
    assert found[0].dispenses["start"].tolist() == [start]


def test_two_way_bootstrap_uses_only_pairs_of_drawn_rows_and_columns() -> None:
    rows = np.array([0, 0, 1, 2])
    columns = np.array([0, 1, 1, 0])

    weights = stats.two_way_weights(rows, columns, 3, 2, np.random.default_rng(1), 400)
    rng = np.random.default_rng(1)
    row_draws = stats.multiplicities(3, rng, 400)
    column_draws = stats.multiplicities(2, rng, 400)

    assert np.array_equal(weights, row_draws[:, rows] * column_draws[:, columns])
    assert np.all((weights[:, 3] > 0) == ((row_draws[:, 2] > 0) & (column_draws[:, 0] > 0)))
    assert row_draws.sum(axis=1).tolist() == [3.0] * 400
