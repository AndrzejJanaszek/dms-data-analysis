import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.api.typing import SeriesGroupBy
from plotly_resampler import FigureWidgetResampler

from dms_analysis.between_events_fragments import NO_FRAGMENT
from dms_analysis.synthetic import ALL, DAY, NIGHT, OTHER, part_of_day

SMOOTHING_S = (10, 30, 60)
THRESHOLDS_KG = (40, 80, 160, 320)
SHORT_SWING_S = (30, 60, 120)
SPANS_S = (10, 20, 30, 45, 60, 90, 120, 180, 300, 600)
DEGREES = {1: "prosta", 2: "parabola"}
PARTS = (NIGHT, OTHER, DAY)
FIGURE_HEIGHT = 500
NO_SPAN = "brak"
BROWSER_HEIGHT = 650


def smoothed_runs(samples: pd.DataFrame, smoothing_s: int) -> pd.DataFrame:
    inside = samples[samples["fragment"] != NO_FRAGMENT]
    values = inside["value"].where(~inside["spike"])
    smoothed = values.groupby(inside["fragment"]).transform(
        lambda fragment: fragment.rolling(smoothing_s, center=True, min_periods=smoothing_s).median()
    )

    runs = pd.DataFrame({"timestamp": inside["timestamp"], "fragment": inside["fragment"], "smoothed": smoothed})
    valid = runs["smoothed"].notna()
    runs["run"] = ((valid != valid.shift()) | (runs["fragment"] != runs["fragment"].shift())).cumsum()
    return runs[valid]


def swings(samples: pd.DataFrame) -> pd.DataFrame:
    tables = []
    for smoothing_s in SMOOTHING_S:
        runs = smoothed_runs(samples, smoothing_s)
        for threshold_kg in THRESHOLDS_KG:
            for _, run in runs.groupby("run"):
                table = _run_swings(run, threshold_kg)
                if not table.empty:
                    tables.append(table.assign(smoothing_s=smoothing_s, threshold_kg=threshold_kg))

    table = pd.concat(tables, ignore_index=True)
    table["part_of_day"] = table["start"].dt.hour.map(part_of_day)
    return table


def swing_summary(swing_table: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    together = pd.concat([swing_table, swing_table.assign(part_of_day=ALL)])
    groups = together.groupby(["smoothing_s", "threshold_kg", "part_of_day"])
    hours = _fragment_hours(samples)

    summary = pd.DataFrame(
        {
            "wahnięć": groups.size(),
            "czas mediana [s]": groups["duration_s"].median(),
            "czas p10 [s]": groups["duration_s"].quantile(0.1),
            "czas p25 [s]": groups["duration_s"].quantile(0.25),
            **{f"< {limit} s [%]": _share_below(groups["duration_s"], limit) for limit in SHORT_SWING_S},
            "zmiana mediana [kg]": groups["amplitude_kg"].median(),
            "zmiana p90 [kg]": groups["amplitude_kg"].quantile(0.9),
        }
    )
    part = summary.index.get_level_values("part_of_day")
    summary.insert(1, "na godzinę", summary["wahnięć"] / part.map(hours).to_numpy())
    return summary.unstack("part_of_day").reindex(columns=[*PARTS, ALL], level=1)


def swing_duration_figure(swing_table: pd.DataFrame, smoothing_s: int, width: int) -> go.Figure:
    figure = go.Figure()
    chosen = swing_table[swing_table["smoothing_s"] == smoothing_s]

    for threshold_kg in THRESHOLDS_KG:
        for part, dash in zip(PARTS, ("dot", "dash", "solid"), strict=True):
            durations = np.sort(
                chosen.loc[(chosen["threshold_kg"] == threshold_kg) & (chosen["part_of_day"] == part), "duration_s"]
            )
            if len(durations) == 0:
                continue
            share = 100 * np.arange(1, len(durations) + 1) / len(durations)
            figure.add_trace(
                go.Scatter(
                    x=durations,
                    y=share,
                    mode="lines",
                    line={"dash": dash},
                    name=f"≥ {threshold_kg} kg, {part} ({len(durations)})",
                    legendgroup=str(threshold_kg),
                )
            )

    figure.update_xaxes(type="log", title_text="czas wahnięcia (od ekstremum do ekstremum) [s]")
    figure.update_yaxes(title_text="wahnięć krótszych [%]", range=[0, 100])
    figure.update_layout(
        title=f"Rozkład skumulowany czasu wahnięć, wygładzanie mediana {smoothing_s} s",
        width=width,
        height=FIGURE_HEIGHT,
    )
    return figure


def span_residuals(samples: pd.DataFrame) -> pd.DataFrame:
    inside = samples[samples["fragment"] != NO_FRAGMENT]
    values = inside["value"].where(~inside["spike"])
    hours = inside["timestamp"].dt.hour
    tables = []

    for span_s in SPANS_S:
        for degree in DEGREES:
            maker = residual_maker(span_s, degree)
            for number, fragment in values.groupby(inside["fragment"]):
                windows = _windows(fragment, span_s)
                complete = ~np.isnan(windows).any(axis=1)
                if not complete.any():
                    continue
                residuals = windows[complete] @ maker.T
                starts = fragment.index[: len(windows) * span_s : span_s][complete]
                tables.append(
                    pd.DataFrame(
                        {
                            "span_s": span_s,
                            "degree": degree,
                            "fragment": number,
                            "start": inside["timestamp"].loc[starts].to_numpy(),
                            "load_kg": windows[complete].mean(axis=1),
                            "std_kg": np.sqrt((residuals**2).sum(axis=1) / (span_s - degree - 1)),
                            "part_of_day": hours.loc[starts].map(part_of_day).to_numpy(),
                        }
                    )
                )

    return pd.concat(tables, ignore_index=True)


def span_fit(values: pd.Series, span_s: int, degree: int) -> pd.Series:
    windows = _windows(values, span_s)
    fitted = np.full(len(values), np.nan)
    fitted[: windows.size] = (windows - windows @ residual_maker(span_s, degree).T).ravel()
    return pd.Series(fitted, index=values.index)


def span_summary(residuals: pd.DataFrame) -> pd.DataFrame:
    together = pd.concat([residuals, residuals.assign(part_of_day=ALL)])
    groups = together.groupby(["degree", "span_s", "part_of_day"])["std_kg"]
    summary = pd.DataFrame(
        {"okien": groups.size(), "mediana [kg]": groups.median(), "p90 [kg]": groups.quantile(0.9)}
    )
    summary = summary.rename(index=DEGREES, level="degree")
    return summary.unstack("part_of_day").reindex(columns=[*PARTS, ALL], level=1)


def span_figure(residuals: pd.DataFrame, width: int) -> go.Figure:
    medians = residuals.groupby(["degree", "part_of_day", "span_s"])["std_kg"].median()
    figure = go.Figure()

    for degree, name in DEGREES.items():
        for part, dash in zip(PARTS, ("dot", "dash", "solid"), strict=True):
            curve = medians.loc[(degree, part)]
            figure.add_trace(
                go.Scatter(
                    x=curve.index,
                    y=curve.to_numpy(),
                    mode="lines+markers",
                    line={"dash": dash},
                    name=f"{name}, {part}",
                    legendgroup=name,
                )
            )

    figure.update_xaxes(type="log", title_text="długość odcinka T [s]")
    figure.update_yaxes(type="log", title_text="odchylenie reszt, mediana z okien [kg]")
    figure.update_layout(
        title="Odchylenie wskazań od prostej i paraboli dopasowanej w odcinku długości T",
        width=width,
        height=FIGURE_HEIGHT,
    )
    return figure


class SwingBrowser:
    def __init__(self, samples: pd.DataFrame, fragments: pd.DataFrame, title: str, width: int) -> None:
        self._samples = samples
        self._fragments = fragments.reset_index(drop=True)
        self._title = title
        self._width = width

        self._smoothing = widgets.Dropdown(
            options=list(SMOOTHING_S), value=SMOOTHING_S[0], description="mediana [s]"
        )
        self._threshold = widgets.Dropdown(
            options=list(THRESHOLDS_KG), value=THRESHOLDS_KG[0], description="próg [kg]"
        )
        self._span = widgets.Dropdown(options=[NO_SPAN, *SPANS_S], value=60, description="odcinek T [s]")
        self._number = widgets.BoundedIntText(
            min=1, max=max(len(self._fragments), 1), description="fragment nr", layout={"width": "180px"}
        )
        self._info = widgets.HTML()
        self._plot = widgets.Box()

        for control in (self._smoothing, self._threshold, self._span, self._number):
            control.observe(lambda _: self._show(), names="value")

        count = widgets.Label(f"z {len(self._fragments)}")
        controls = widgets.HBox([self._smoothing, self._threshold, self._span, self._number, count])
        self.widget = widgets.VBox([controls, self._info, self._plot])
        self._show()

    def _show(self) -> None:
        fragment = self._fragments.iloc[self._number.value - 1]
        shown = self._samples[self._samples["fragment"] == fragment["fragment"]]
        runs = smoothed_runs(shown, self._smoothing.value)
        found = [_run_swings(run, self._threshold.value) for _, run in runs.groupby("run")]
        swing_table = pd.concat(found, ignore_index=True) if found else pd.DataFrame()

        values = shown["value"].where(~shown["spike"])
        fits = {} if self._span.value == NO_SPAN else {
            degree: span_fit(values, self._span.value, degree) for degree in DEGREES
        }

        durations = swing_table["duration_s"] if not swing_table.empty else pd.Series(dtype=float)
        self._info.value = (
            f"fragment {fragment['fragment']}, {fragment['start']:%Y-%m-%d %H:%M:%S},"
            f" {fragment['duration_min']:.0f} min | wahnięć {len(durations)},"
            f" czas mediana {durations.median():.0f} s, krótszych niż 60 s {(durations < 60).sum()}"
            f"{self._fit_info(values, fits)}"
        )
        for previous in self._plot.children:
            previous.close()
        self._plot.children = [self._figure(shown, runs, swing_table, fits)]

    def _fit_info(self, values: pd.Series, fits: dict[int, pd.Series]) -> str:
        if not fits:
            return ""

        span_s = self._span.value
        window = np.arange(len(values)) // span_s
        parts = []
        for degree, fit in fits.items():
            squares = ((values - fit) ** 2).groupby(window).sum(min_count=span_s)
            std = np.sqrt(squares / (span_s - degree - 1))
            parts.append(f"{DEGREES[degree]} {std.median():.0f} kg")
        return f" | odchylenie reszt w odcinkach {span_s} s, mediana: {', '.join(parts)}"

    def _figure(
        self, shown: pd.DataFrame, runs: pd.DataFrame, swing_table: pd.DataFrame, fits: dict[int, pd.Series]
    ) -> FigureWidgetResampler:
        figure = FigureWidgetResampler(go.Figure())
        normal = shown[~shown["spike"]]

        figure.add_trace(
            go.Scatter(name="wskazanie", mode="lines+markers", marker={"size": 3}, line={"width": 1}),
            hf_x=normal["timestamp"],
            hf_y=normal["value"],
        )
        figure.add_trace(
            go.Scatter(name=f"mediana {self._smoothing.value} s", mode="lines", line={"width": 2}),
            hf_x=runs["timestamp"],
            hf_y=runs["smoothed"],
        )
        if not swing_table.empty:
            times, levels = _zigzag_line(swing_table)
            figure.add_trace(
                go.Scatter(
                    name=f"wahnięcia ≥ {self._threshold.value} kg",
                    mode="lines+markers",
                    marker={"size": 7, "color": "red"},
                    line={"width": 2, "color": "red"},
                ),
                hf_x=times,
                hf_y=levels,
            )
        for degree, fit in fits.items():
            figure.add_trace(
                go.Scatter(name=f"{DEGREES[degree]}, odcinki {self._span.value} s", mode="lines", line={"width": 2}),
                hf_x=shown["timestamp"],
                hf_y=fit,
            )

        figure.update_yaxes(title_text="wskazanie [kg]")
        figure.update_layout(
            title=self._title,
            height=BROWSER_HEIGHT,
            width=self._width,
            autosize=False,
            hovermode="x unified",
            margin={"l": 60, "r": 20, "t": 60, "b": 40},
        )
        return figure


def _run_swings(run: pd.DataFrame, threshold_kg: float) -> pd.DataFrame:
    values = run["smoothed"].to_numpy()
    extremes = _zigzag(values, threshold_kg)[1:]
    if len(extremes) < 2:
        return pd.DataFrame()

    times = run["timestamp"].iloc[extremes]
    starts = times.iloc[:-1].reset_index(drop=True)
    ends = times.iloc[1:].reset_index(drop=True)
    levels = values[extremes]
    return pd.DataFrame(
        {
            "fragment": run["fragment"].iloc[0],
            "start": starts,
            "end": ends,
            "duration_s": (ends - starts).dt.total_seconds(),
            "start_kg": levels[:-1],
            "end_kg": levels[1:],
            "amplitude_kg": np.abs(np.diff(levels)),
        }
    )


def _zigzag(values: np.ndarray, threshold_kg: float) -> list[int]:
    extremes: list[int] = []
    direction = 0
    low = high = candidate = 0

    for position, value in enumerate(values):
        if direction == 0:
            if value < values[low]:
                low = position
            if value > values[high]:
                high = position
            if value - values[low] >= threshold_kg:
                extremes.append(low)
                direction, candidate = 1, position
            elif values[high] - value >= threshold_kg:
                extremes.append(high)
                direction, candidate = -1, position
        elif direction == 1:
            if value > values[candidate]:
                candidate = position
            elif values[candidate] - value >= threshold_kg:
                extremes.append(candidate)
                direction, candidate = -1, position
        else:
            if value < values[candidate]:
                candidate = position
            elif value - values[candidate] >= threshold_kg:
                extremes.append(candidate)
                direction, candidate = 1, position

    return extremes


def _zigzag_line(swing_table: pd.DataFrame) -> tuple[list[object], list[float]]:
    times: list[object] = []
    levels: list[float] = []
    previous_end = None

    for swing in swing_table.itertuples():
        if swing.start != previous_end:
            times += [swing.start, swing.start]
            levels += [np.nan, swing.start_kg]
        times.append(swing.end)
        levels.append(swing.end_kg)
        previous_end = swing.end

    return times[1:], levels[1:]


def _share_below(durations: SeriesGroupBy, limit_s: int) -> pd.Series:
    return durations.apply(lambda values: 100 * (values < limit_s).mean())


def _windows(values: pd.Series, span_s: int) -> np.ndarray:
    count = len(values) // span_s
    return values.to_numpy()[: count * span_s].reshape(count, span_s)


def residual_maker(span_s: int, degree: int) -> np.ndarray:
    seconds = np.linspace(-1.0, 1.0, span_s)
    design = np.vander(seconds, degree + 1, increasing=True)
    return np.eye(span_s) - design @ np.linalg.pinv(design)


def _fragment_hours(samples: pd.DataFrame) -> dict[str, float]:
    inside = samples[samples["fragment"] != NO_FRAGMENT]
    parts = inside["timestamp"].dt.hour.map(part_of_day)
    hours = (parts.value_counts() / 3600).to_dict()
    hours[ALL] = len(inside) / 3600
    return hours
