import numpy as np
import pandas as pd
from event_helpers import events_frame, period_data

from dms_analysis.errors import stats
from dms_analysis.errors.fragments import CLEARANCE_S, free_intervals, real_events
from dms_analysis.errors.grid import grid_starts, period_grid, start_errors
from dms_analysis.events.values import DISPENSE, REFILL, WINDOW_S, Weight, dispense_masses

START = pd.Timestamp("2026-11-03 00:00", tz="Europe/Warsaw").timestamp()


def noisy_weight(seconds_count: int = 4000, seed: int = 1) -> Weight:
    rng = np.random.default_rng(seed)
    seconds = START + np.cumsum(1.0 + rng.uniform(0.0, 0.002, seconds_count))
    seconds[rng.integers(0, seconds_count, 20)] += 0.6
    values = np.round((40000.0 + np.cumsum(rng.normal(0.0, 8.0, seconds_count))) / 20.0) * 20.0
    return Weight(np.sort(seconds), values)


def test_vectorised_errors_match_dispense_masses() -> None:
    weight = noisy_weight()
    rng = np.random.default_rng(2)
    starts = np.floor(rng.uniform(weight.seconds[10], weight.seconds[-200], 300))

    errors, usable = start_errors(weight, starts, 30.0)

    for position, start in enumerate(starts[usable]):
        for row in dispense_masses(weight, start, start + 30.0):
            expected = round(float(row["mass_kg"]), 6)
            assert errors[(str(row["method"]), float(row["delay_s"]))][position] == expected


def test_starts_skip_windows_with_too_few_samples() -> None:
    seconds = START + np.arange(0.0, 400.0)
    seconds = np.delete(seconds, [101, 102])
    weight = Weight(seconds, np.full(len(seconds), 40000.0))

    errors, usable = start_errors(weight, np.array([START + 50.0, START + 103.0, START + 300.0]), 10.0)

    assert usable.tolist() == [True, False, True]
    assert all(len(values) == 2 for values in errors.values())


def test_grid_starts_keep_all_windows_inside_fragment() -> None:
    starts = grid_starts(np.array([[100.2, 200.0]]), 30.0)

    assert starts[0] == np.ceil(100.2 + WINDOW_S)
    assert starts[-1] + 30.0 + 3.0 + WINDOW_S <= 200.0
    assert np.all(np.diff(starts) == 1.0)


def test_grid_on_flat_weight_has_zero_error_everywhere() -> None:
    seconds = START + np.arange(0.0, 7200.0)
    weight = Weight(seconds, np.full(len(seconds), 40000.0))

    result = period_grid(weight, np.array([[seconds[0], seconds[-1]]]))

    assert set(result.counts["error_kg"]) == {0.0}
    assert result.skipped.empty
    assert set(result.counts["hour"]) == {0, 1}


def test_measures_from_counts_equal_measures_from_raw_errors() -> None:
    rng = np.random.default_rng(3)
    raw = np.round(rng.normal(5.0, 30.0, 5000) / 20.0) * 20.0
    values, weights = np.unique(raw, return_counts=True)

    result = stats.measures(values, weights)

    assert np.isclose(result[stats.MEAN][0], raw.mean())
    assert np.isclose(result[stats.STD][0], raw.std(ddof=1))
    assert np.isclose(result[stats.RMSE][0], np.sqrt(np.mean(raw**2)))
    assert np.isclose(result[stats.MAE][0], np.mean(np.abs(raw)))
    for name, percentile in stats.PERCENTILES.items():
        assert result[name][0] == np.percentile(np.abs(raw), percentile, method="inverted_cdf")
    assert result[stats.MAX][0] == np.abs(raw).max()


def test_bootstrap_draws_whole_days_and_repeats_with_seed() -> None:
    frame = pd.DataFrame(
        {
            "unit": ["2026-11-03", "2026-11-03", "2026-11-04", "2026-11-05"],
            "error_kg": [0.0, 20.0, 40.0, 60.0],
            "starts": [7, 3, 5, 11],
        }
    )
    _, _, histograms = stats.unit_histograms(frame)

    weights = stats.resampled_weights(histograms, np.random.default_rng(4), 500)

    assert np.all(weights[:, 0] % 7 == 0)
    assert np.all(weights[:, 0] / 7 == weights[:, 1] / 3)
    assert np.all(weights.sum(axis=1) >= 3 * 5)
    first = stats.summary(frame, np.random.default_rng(5), 200)
    second = stats.summary(frame, np.random.default_rng(5), 200)
    pd.testing.assert_frame_equal(first, second)
    assert first.set_index("measure").loc[stats.MAX, "max_day"] == "2026-11-05"


def test_real_events_add_review_accepted_and_drop_false_events() -> None:
    events = events_frame(
        [
            (DISPENSE, START + 100.0, START + 130.0, 0, 0),
            (DISPENSE, START + 500.0, START + 505.0, 1, 0),
            (DISPENSE, START + 900.0, START + 904.0, 1, 0),
            (DISPENSE, START + 1300.0, START + 1310.0, 0, 0),
            (REFILL, START + 2000.0, START + 2600.0, 0, 1),
        ]
    )
    issues = pd.DataFrame(
        [("dispense", "false_event", START + 1299.0, START + 1311.0, "")],
        columns=["channel", "kind", "start", "end", "note"],
    )
    data = period_data(noisy_weight(), events, issues)
    good = pd.Timestamp(START + 500.0, unit="s", tz="UTC").tz_convert("Europe/Warsaw").isoformat()
    marks = pd.DataFrame([{"period": data.full_id, "start": good, "end": good, "mark": "dobry"}])

    real = real_events(data, marks)

    assert (real["start"] - START).tolist() == [100.0, 500.0, 2000.0]


def test_free_intervals_keep_clear_of_real_events() -> None:
    events = events_frame([(DISPENSE, START + 1000.0, START + 1030.0, 0, 0)])
    data = period_data(noisy_weight(), events)

    free = free_intervals(data, real_events(data, pd.DataFrame(columns=["period", "start", "end", "mark"])))

    assert np.any(np.isclose(free[:, 1], START + 1000.0 - CLEARANCE_S))
    assert np.any(np.isclose(free[:, 0], START + 1030.0 + CLEARANCE_S))


def test_worst_starts_are_separated_episodes() -> None:
    seconds = START + np.arange(0.0, 7200.0)
    values = np.full(len(seconds), 40000.0)
    values[1000:1100] -= 400.0
    values[5000:5050] -= 200.0
    weight = Weight(seconds, values)

    worst = period_grid(weight, np.array([[seconds[0], seconds[-1]]])).worst
    chosen = np.sort(worst[worst["duration_s"] == 30]["start"].to_numpy())

    assert np.all(np.diff(chosen) >= 120.0)
    assert worst[worst["duration_s"] == 30]["error_kg"].abs().max() == 400.0
