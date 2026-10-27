from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dms_analysis.between_events_fragments import DAY_HOURS, NIGHT_HOURS, NO_FRAGMENT, RESOLUTION_KG
from dms_analysis.event_stats import event_differences
from dms_analysis.prepared import PreparedSession

DISPENSE = "odsyp"
SOURCE_METHOD = "mediana 3 s"
TRIALS = 2000
TRIAL_SEED = 0
DRIFT_WINDOWS_S = (10, 20, 30, 60)
SETTLING_S = 3
MARGIN_S = max(DRIFT_WINDOWS_S) + SETTLING_S + 7
NIGHT = "noc"
DAY = "dzień"
OTHER = "pozostałe"
ALL = "wszystkie"
FIGURE_HEIGHT = 500
WINDOW_COLOR = "rgba(0, 160, 0, 0.15)"


@dataclass(frozen=True)
class Method:
    statistic: str | None
    window_s: float | None
    degree: int | None = None


METHODS = {
    "najbliższa próbka": Method(None, None),
    "mediana 3 s": Method("median", 3),
    "mediana 5 s": Method("median", 5),
    "mediana 10 s": Method("median", 10),
    "średnia 3 s": Method("mean", 3),
    "średnia 5 s": Method("mean", 5),
    "średnia 10 s": Method("mean", 10),
    **{f"prosta + skok {window} s": Method(None, window, 1) for window in DRIFT_WINDOWS_S},
    **{f"wielomian 3 + skok {window} s": Method(None, window, 3) for window in DRIFT_WINDOWS_S},
}


def real_dispenses(session: PreparedSession) -> pd.DataFrame:
    table = event_differences(session)
    chosen = (table["role"] == DISPENSE) & (table["method"] == SOURCE_METHOD) & (table["difference_kg"] < 0)
    dispenses = table[chosen]
    return pd.DataFrame(
        {
            "mass_kg": -dispenses["difference_kg"],
            "duration_s": dispenses["duration_s"],
            "hour": dispenses["start"].dt.hour,
        }
    ).reset_index(drop=True)


def draw_trials(samples: pd.DataFrame, dispenses: pd.DataFrame, same_hour: bool = False) -> pd.DataFrame:
    inside = samples[samples["fragment"] != NO_FRAGMENT]
    groups = inside.groupby("fragment")["timestamp"]
    since_start = (inside["timestamp"] - groups.transform("first")).dt.total_seconds().to_numpy()
    to_end = (groups.transform("last") - inside["timestamp"]).dt.total_seconds().to_numpy()
    hours = inside["timestamp"].dt.hour.to_numpy()

    generator = np.random.default_rng(TRIAL_SEED)
    rows = []
    for _ in range(TRIALS):
        source = dispenses.iloc[generator.integers(len(dispenses))]
        fitting = (since_start >= MARGIN_S) & (to_end >= source["duration_s"] + MARGIN_S)
        if same_hour:
            fitting &= hours == source["hour"]
        fits = np.flatnonzero(fitting)
        if len(fits) == 0:
            continue
        position = fits[generator.integers(len(fits))]
        rows.append(
            {
                "fragment": inside["fragment"].iloc[position],
                "start": inside["timestamp"].iloc[position],
                "duration_s": source["duration_s"],
                "mass_kg": source["mass_kg"],
            }
        )

    trials = pd.DataFrame(rows)
    trials["end"] = trials["start"] + pd.to_timedelta(trials["duration_s"], unit="s")
    trials["part_of_day"] = trials["start"].dt.hour.map(part_of_day)
    return trials


def synthetic_segment(samples: pd.DataFrame, trial: pd.Series) -> pd.DataFrame:
    low = trial["start"] - pd.Timedelta(seconds=MARGIN_S)
    high = trial["end"] + pd.Timedelta(seconds=MARGIN_S)
    segment = samples.loc[samples["timestamp"].between(low, high), ["timestamp", "value", "spike"]].copy()

    progress = ((segment["timestamp"] - trial["start"]).dt.total_seconds() / trial["duration_s"]).clip(0, 1)
    segment["synthetic"] = np.round((segment["value"] - trial["mass_kg"] * progress) / RESOLUTION_KG) * RESOLUTION_KG
    return segment


def measure(samples: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for number, trial in trials.iterrows():
        segment = synthetic_segment(samples, trial)
        if segment["spike"].any():
            continue
        weight = segment.set_index("timestamp")["synthetic"]
        for name, method in METHODS.items():
            measured = _mass(weight, trial["start"], trial["end"], method)
            rows.append(
                {
                    "trial": number,
                    "method": name,
                    "part_of_day": trial["part_of_day"],
                    "mass_kg": trial["mass_kg"],
                    "measured_kg": measured,
                    "error_kg": measured - trial["mass_kg"],
                }
            )
    return pd.DataFrame(rows)


def error_summary(results: pd.DataFrame) -> pd.DataFrame:
    together = pd.concat([results, results.assign(part_of_day=ALL)])
    groups = together.groupby(["method", "part_of_day"], sort=False)["error_kg"]
    summary = pd.DataFrame(
        {
            "prób": groups.size(),
            "błąd średni [kg]": groups.mean(),
            "odchylenie std [kg]": groups.std(),
            "|błąd| mediana [kg]": groups.apply(lambda errors: errors.abs().median()),
            "|błąd| p95 [kg]": groups.apply(lambda errors: errors.abs().quantile(0.95)),
            "p5 [kg]": groups.quantile(0.05),
            "p95 [kg]": groups.quantile(0.95),
        }
    )
    return summary.unstack("part_of_day").reindex(list(METHODS)).reindex(columns=[NIGHT, DAY, OTHER, ALL], level=1)


def error_figure(results: pd.DataFrame, width: int) -> go.Figure:
    figure = go.Figure()
    for part in (NIGHT, DAY, OTHER):
        chosen = results[results["part_of_day"] == part]
        figure.add_trace(go.Box(x=chosen["method"], y=chosen["error_kg"], name=part, boxpoints=False))

    figure.update_layout(
        title="Błąd wyliczonej masy sztucznego odsypu (wąsy: 1,5 rozstępu międzykwartylowego)",
        xaxis_title="metoda",
        yaxis_title="wyliczona − prawdziwa masa [kg]",
        boxmode="group",
        width=width,
        height=FIGURE_HEIGHT,
    )
    return figure


def worst_trials(results: pd.DataFrame) -> pd.Index:
    return results.groupby("trial")["error_kg"].apply(lambda errors: errors.abs().max()).sort_values(ascending=False).index


def trial_figure(
    samples: pd.DataFrame, trials: pd.DataFrame, results: pd.DataFrame, number: int, width: int
) -> go.Figure:
    trial = trials.loc[number]
    segment = synthetic_segment(samples, trial)
    errors = results[results["trial"] == number].set_index("method")["error_kg"]
    description = ", ".join(f"{name} {error:+.0f}" for name, error in errors.items())

    figure = go.Figure(
        [
            go.Scatter(
                name="prawdziwe wskazania",
                x=segment["timestamp"],
                y=segment["value"],
                mode="lines+markers",
                marker={"size": 3},
            ),
            go.Scatter(
                name="ze sztucznym odsypem",
                x=segment["timestamp"],
                y=segment["synthetic"],
                mode="lines+markers",
                marker={"size": 3},
            ),
        ]
    )
    figure.add_vrect(x0=trial["start"], x1=trial["end"], fillcolor=WINDOW_COLOR, line_width=0)
    figure.update_layout(
        title=(
            f"Próba {number}: {trial['start']:%Y-%m-%d %H:%M:%S}, {trial['mass_kg']:.0f} kg w {trial['duration_s']:.0f} s,"
            f" {trial['part_of_day']}<br><sub>błąd [kg]: {description}</sub>"
        ),
        yaxis_title="wskazanie [kg]",
        width=width,
        height=FIGURE_HEIGHT,
    )
    return figure


def _mass(weight: pd.Series, start: pd.Timestamp, end: pd.Timestamp, method: Method) -> float:
    if method.window_s is None:
        positions = weight.index.get_indexer(pd.DatetimeIndex([start, end]), method="nearest")
        before, after = weight.iloc[positions]
        return float(before - after)

    window = pd.Timedelta(seconds=method.window_s)
    if method.degree is not None:
        return _step_fit(weight, start, end, window, method.degree)

    before = weight[(weight.index >= start - window) & (weight.index < start)]
    after = weight[(weight.index >= end) & (weight.index < end + window)]
    return float(before.agg(method.statistic) - after.agg(method.statistic))


def _step_fit(weight: pd.Series, start: pd.Timestamp, end: pd.Timestamp, window: pd.Timedelta, degree: int) -> float:
    settled = end + pd.Timedelta(seconds=SETTLING_S)
    before = weight[(weight.index >= start - window) & (weight.index < start)]
    after = weight[(weight.index >= settled) & (weight.index < settled + window)]

    middle = start + (settled - start) / 2
    half_span = ((settled + window) - middle).total_seconds()
    times = np.concatenate([(before.index - middle).total_seconds(), (after.index - middle).total_seconds()]) / half_span
    step = np.concatenate([np.zeros(len(before)), np.ones(len(after))])

    design = np.column_stack([np.vander(times, degree + 1, increasing=True), step])
    values = np.concatenate([before.to_numpy(), after.to_numpy()])
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return float(-coefficients[-1])


def part_of_day(hour: int) -> str:
    if hour in NIGHT_HOURS:
        return NIGHT
    if hour in DAY_HOURS:
        return DAY
    return OTHER
