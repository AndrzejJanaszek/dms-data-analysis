import numpy as np
import pandas as pd
import pytest

from event_helpers import events_frame, period_data, weight_from

from dms_analysis.events.values import (
    DISPENSE,
    EXCLUDED_COLLISION,
    EXCLUDED_WINDOW,
    MEAN,
    MEDIAN,
    NEAREST,
    REFILL,
    Weight,
    dispense_masses,
    dispense_tables,
    refill_table,
)


def masses_by_variant(weight: Weight, start: float, end: float) -> dict[tuple[str, float], float]:
    return {(row["method"], row["delay_s"]): row["mass_kg"] for row in dispense_masses(weight, start, end)}


def test_methods_use_three_samples_before_start_and_after_end_plus_delay() -> None:
    weight = weight_from([1000, 1020, 1040, 900, 800, 700, 600, 500, 480, 460, 440, 420])

    masses = masses_by_variant(weight, 3.0, 5.0)

    assert masses[(NEAREST, 0.0)] == 1040 - 700
    assert masses[(MEDIAN, 0.0)] == 1020 - 600
    assert masses[(MEAN, 0.0)] == pytest.approx(1020 - 600)
    assert masses[(NEAREST, 3.0)] == 1040 - 480
    assert masses[(MEDIAN, 3.0)] == 1020 - 460
    assert masses[(MEAN, 2.0)] == pytest.approx(1020 - (500 + 480 + 460) / 3)


def test_no_drop_is_flagged_per_variant() -> None:
    weight = weight_from([1000] * 12)

    rows = dispense_masses(weight, 3.0, 5.0)

    assert all(row["no_drop"] for row in rows)


def test_rejected_dispense_is_not_a_neighbour() -> None:
    weight = weight_from([1000.0] * 60)
    events = events_frame([(DISPENSE, 20.0, 30.0, 0, 0), (DISPENSE, 31.0, 32.0, 1, 0)])

    dispenses, _ = dispense_tables(period_data(weight, events))

    assert dispenses["excluded"].tolist() == [""]


@pytest.mark.parametrize("neighbour_start", [33.0, 15.5])
def test_neighbour_inside_window_excludes_dispense(neighbour_start: float) -> None:
    weight = weight_from([1000.0] * 60)
    events = events_frame([(DISPENSE, 20.0, 28.0, 0, 0), (DISPENSE, neighbour_start, neighbour_start + 2.0, 0, 0)])

    dispenses, _ = dispense_tables(period_data(weight, events))

    assert dispenses.set_index("start").loc[20.0, "excluded"] == EXCLUDED_WINDOW


def test_issue_gap_and_part_boundary_exclude_dispense() -> None:
    weight = weight_from([1000.0] * 120)
    events = events_frame([(DISPENSE, 20.0, 28.0, 0, 0), (DISPENSE, 60.0, 68.0, 0, 0), (DISPENSE, 95.0, 100.0, 0, 0)])
    issues = pd.DataFrame({"channel": ["weight"], "kind": ["disturbance"], "start": [30.0], "end": [40.0], "note": [""]})
    parts = pd.DataFrame({"part": [1, 2], "start": [0.0, 70.0], "end": [66.0, 119.0]})
    gapped = Weight(np.delete(weight.seconds, [97]), np.delete(weight.values, [97]))

    dispenses, _ = dispense_tables(period_data(gapped, events, issues, parts))

    assert dispenses["excluded"].tolist() == [EXCLUDED_WINDOW, EXCLUDED_WINDOW, ""]


def test_missing_sample_in_window_excludes_dispense() -> None:
    weight = weight_from([1000.0] * 60)
    gapped = Weight(np.delete(weight.seconds, [32]), np.delete(weight.values, [32]))
    events = events_frame([(DISPENSE, 20.0, 28.0, 0, 0)])

    dispenses, _ = dispense_tables(period_data(gapped, events))

    assert dispenses["excluded"].tolist() == [EXCLUDED_WINDOW]


def test_dispense_overlapping_manual_refill_is_collision() -> None:
    weight = weight_from([1000.0] * 200)
    events = events_frame([(REFILL, 10.0, 150.0, 0, 1), (DISPENSE, 40.0, 50.0, 0, 0), (DISPENSE, 152.0, 160.0, 0, 0)])

    dispenses, _ = dispense_tables(period_data(weight, events))

    assert dispenses["excluded"].tolist() == [EXCLUDED_COLLISION, EXCLUDED_COLLISION]


def test_gaps_to_neighbours() -> None:
    weight = weight_from([1000.0] * 200)
    events = events_frame([(DISPENSE, 10.0, 20.0, 0, 0), (DISPENSE, 60.0, 70.0, 0, 0), (REFILL, 100.0, 150.0, 0, 1)])

    dispenses, _ = dispense_tables(period_data(weight, events))

    assert dispenses["gap_before_s"].isna().iloc[0]
    assert dispenses["gap_after_s"].tolist() == [40.0, 30.0]


def test_refill_mass_and_dispenses_inside() -> None:
    values = [1000.0] * 10 + [1000.0 + 100 * step for step in range(1, 41)] + [5000.0] * 10
    weight = weight_from(values)
    events = events_frame([(REFILL, 8.0, 52.0, 0, 1), (DISPENSE, 20.0, 25.0, 0, 0)])

    refills = refill_table(period_data(weight, events))

    assert refills.loc[0, "mass_nearest_kg"] == 4000.0
    assert refills.loc[0, "mass_median_kg"] == 4000.0
    assert refills.loc[0, "dispenses_inside"] == 1
