from collections.abc import Sequence

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from plotly_resampler import FigureWidgetResampler

from dms_analysis.prepared import PreparedSession

MARGIN = pd.Timedelta(seconds=30)
MIN_DURATION = pd.Timedelta(minutes=5)
MAX_SAMPLE_GAP = pd.Timedelta(seconds=5)
SPIKE_WINDOW_SAMPLES = 121
SPIKE_THRESHOLD_KG = 1000.0
WINDOWS_S = (5, 10, 30, 60, 300)
ALLAN_TAUS_S = (1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 1800, 3600)
RESOLUTION_KG = 20.0
WHITE_NOISE_SEED = 0
MIN_ALLAN_AVERAGES = 10
NIGHT_HOURS = (*range(22, 24), *range(0, 10))
DAY_HOURS = tuple(range(13, 20))
DAY_SHARE_COLORSCALE = "Viridis"
NO_FRAGMENT = -1
SECONDS_PER_HOUR = 3600
FIGURE_HEIGHT = 450
BROWSER_HEIGHT = 900
DEFAULT_WINDOW_S = 60
DEFAULT_PADDING_S = 60.0
EVENT_COLOR = "rgba(0, 160, 0, 0.15)"
REJECTED_COLOR = "rgba(220, 0, 0, 0.15)"

IN_FRAGMENT = "we fragmencie"
NEAR_EVENT = "w zdarzeniu albo marginesie"
OUTSIDE = "poza fragmentem"


def prepare_samples(session: PreparedSession) -> pd.DataFrame:
    weight = session.data.weight
    samples = pd.DataFrame({"timestamp": weight["timestamp"], "value": weight["value"].astype(float)})
    samples["busy"] = _busy(samples["timestamp"], session.events)
    samples["fragment"] = _fragments(samples)

    median = samples["value"].rolling(SPIKE_WINDOW_SAMPLES, center=True, min_periods=1).median()
    samples["median_kg"] = median
    samples["deviation_kg"] = samples["value"] - median
    samples["spike"] = samples["deviation_kg"].abs() > SPIKE_THRESHOLD_KG

    for window_s in WINDOWS_S:
        slope, std = _rolling_line(samples, window_s)
        samples[slope_column(window_s)] = slope
        samples[std_column(window_s)] = std

    return samples


def slope_column(window_s: int) -> str:
    return f"slope_kg_h_{window_s}"


def std_column(window_s: int) -> str:
    return f"std_kg_{window_s}"


def fragment_table(samples: pd.DataFrame) -> pd.DataFrame:
    inside = samples[samples["fragment"] != NO_FRAGMENT]
    groups = inside.groupby("fragment")

    table = pd.DataFrame(
        {
            "start": groups["timestamp"].first(),
            "end": groups["timestamp"].last(),
            "samples": groups.size(),
            "spike_samples": groups["spike"].sum(),
            "first_kg": groups["value"].first(),
            "last_kg": groups["value"].last(),
            "min_kg": groups["value"].min(),
            "max_kg": groups["value"].max(),
        }
    )
    table["duration_min"] = (table["end"] - table["start"]).dt.total_seconds() / 60
    table["change_kg"] = table["last_kg"] - table["first_kg"]
    table["change_kg_h"] = table["change_kg"] / table["duration_min"] * 60
    return table.reset_index()


def spike_episodes(samples: pd.DataFrame) -> pd.DataFrame:
    spikes = samples[samples["spike"]]
    if spikes.empty:
        return pd.DataFrame(columns=["start", "end", "samples", "extreme_deviation_kg", "place"])

    episode = (spikes.index.to_series().diff() != 1).cumsum()
    groups = spikes.groupby(episode)
    extreme = groups["deviation_kg"].agg(lambda values: values.loc[values.abs().idxmax()])

    return pd.DataFrame(
        {
            "start": groups["timestamp"].first(),
            "end": groups["timestamp"].last(),
            "samples": groups.size(),
            "extreme_deviation_kg": extreme,
            "place": groups.apply(_place, include_groups=False),
        }
    ).reset_index(drop=True)


def window_summary(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window_s in WINDOWS_S:
        std = samples[std_column(window_s)].dropna()
        slope = samples[slope_column(window_s)].abs().dropna()
        rows.append(
            {
                "okno [s]": window_s,
                "okien": len(std),
                "odchylenie = 0 [%]": 100 * (std == 0).mean(),
                "odchylenie mediana [kg]": std.median(),
                "odchylenie p90 [kg]": std.quantile(0.9),
                "odchylenie p99 [kg]": std.quantile(0.99),
                "|nachylenie| mediana [kg/h]": slope.median(),
                "|nachylenie| p90 [kg/h]": slope.quantile(0.9),
                "|nachylenie| p99 [kg/h]": slope.quantile(0.99),
            }
        )
    return pd.DataFrame(rows).set_index("okno [s]")


def window_box_figure(samples: pd.DataFrame, column: str, title: str, y_title: str, width: int) -> go.Figure:
    figure = go.Figure()
    for window_s in WINDOWS_S:
        values = samples[f"{column}_{window_s}"].dropna()
        if column.startswith("slope"):
            values = values.abs()
        q05, q25, q50, q75, q95 = values.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        figure.add_trace(
            go.Box(
                name=f"{window_s} s",
                q1=[q25],
                median=[q50],
                q3=[q75],
                lowerfence=[q05],
                upperfence=[q95],
                x=[f"{window_s} s"],
            )
        )

    figure.update_layout(
        title=f"{title} (wąsy: 5. i 95. percentyl)",
        xaxis_title="okno",
        yaxis_title=y_title,
        showlegend=False,
        width=width,
        height=FIGURE_HEIGHT,
    )
    return figure


def allan_deviation(samples: pd.DataFrame, hours: Sequence[int] | None = None) -> pd.DataFrame:
    inside = samples[samples["fragment"] != NO_FRAGMENT]
    values = inside["value"].where(~inside["spike"])
    if hours is not None:
        values = values.where(inside["timestamp"].dt.hour.isin(hours))
    rows = []

    for tau in ALLAN_TAUS_S:
        squares = [_squared_differences(fragment, tau) for _, fragment in values.groupby(inside["fragment"])]
        pooled = np.concatenate(squares)
        deviation = float(np.sqrt(0.5 * pooled.mean())) if len(pooled) else np.nan
        rows.append({"tau_s": tau, "allan_kg": deviation, "pairs": len(pooled)})

    return pd.DataFrame(rows)


def fragment_allan(samples: pd.DataFrame, fragments: pd.DataFrame, day_hours: Sequence[int]) -> pd.DataFrame:
    inside = samples[samples["fragment"] != NO_FRAGMENT]
    rows = []

    for number, fragment in inside.groupby("fragment"):
        values = fragment["value"].where(~fragment["spike"])
        row: dict[str, object] = {
            "fragment": number,
            "day_share_pct": 100 * fragment["timestamp"].dt.hour.isin(day_hours).mean(),
        }
        for tau in ALLAN_TAUS_S:
            reliable = len(fragment) >= MIN_ALLAN_AVERAGES * tau
            squares = _squared_differences(values, tau)
            row[tau] = float(np.sqrt(0.5 * squares.mean())) if reliable and len(squares) else np.nan
        rows.append(row)

    table = fragments[["fragment", "start", "duration_min"]].merge(pd.DataFrame(rows), on="fragment")
    curves = table[list(ALLAN_TAUS_S)]
    table["min_allan_kg"] = curves.min(axis=1)
    table["tau_at_min_s"] = curves.idxmin(axis=1)
    table["min_to_1s"] = table["min_allan_kg"] / table[ALLAN_TAUS_S[0]]
    return table


def white_noise_samples(samples: pd.DataFrame, sigma_kg: float) -> pd.DataFrame:
    generator = np.random.default_rng(WHITE_NOISE_SEED)
    noise = generator.normal(0.0, sigma_kg, len(samples))

    noisy = samples.copy()
    noisy["value"] = np.round((samples["value"].mean() + noise) / RESOLUTION_KG) * RESOLUTION_KG
    return noisy


def allan_figure(curves: dict[str, pd.DataFrame], width: int) -> go.Figure:
    figure = go.Figure()
    for name, allan in curves.items():
        figure.add_trace(go.Scatter(name=name, x=allan["tau_s"], y=allan["allan_kg"], mode="lines+markers"))

    _allan_axes(figure, "Odchylenie Allana wskazań we fragmentach bez zdarzeń", width)
    return figure


def fragment_allan_figure(table: pd.DataFrame, white: pd.DataFrame, width: int) -> go.Figure:
    figure = go.Figure()
    colors = sample_colorscale(DAY_SHARE_COLORSCALE, list(table["day_share_pct"] / 100))
    taus = list(ALLAN_TAUS_S)

    for (_, row), color in zip(table.iterrows(), colors, strict=True):
        label = (
            f"fragment {row['fragment']}, {row['start']:%Y-%m-%d %H:%M}, {row['duration_min']:.0f} min,"
            f" w dzień {row['day_share_pct']:.0f}%"
        )
        figure.add_trace(
            go.Scatter(
                x=taus,
                y=row[taus].tolist(),
                mode="lines+markers",
                marker={"size": 4, "color": color},
                line={"width": 1, "color": color},
                name=label,
                hovertemplate=f"{label}<br>τ %{{x}} s: %{{y:.1f}} kg<extra></extra>",
                showlegend=False,
            )
        )

    figure.add_trace(
        go.Scatter(
            x=white["tau_s"],
            y=white["allan_kg"],
            mode="lines",
            line={"width": 2, "dash": "dash", "color": "black"},
            name="biały szum",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={
                "colorscale": DAY_SHARE_COLORSCALE,
                "cmin": 0,
                "cmax": 100,
                "color": [0],
                "colorbar": {"title": "w dzień [%]"},
            },
            showlegend=False,
            hoverinfo="skip",
        )
    )

    _allan_axes(figure, "Odchylenie Allana osobno dla każdego fragmentu", width)
    figure.update_layout(height=BROWSER_HEIGHT)
    return figure


def _allan_axes(figure: go.Figure, title: str, width: int) -> None:
    figure.update_xaxes(type="log", title_text="czas uśredniania τ [s]")
    figure.update_yaxes(type="log", title_text="odchylenie Allana [kg]")
    figure.update_layout(title=title, width=width, height=FIGURE_HEIGHT)


def _squared_differences(values: pd.Series, tau: int) -> np.ndarray:
    averages = values.reset_index(drop=True).rolling(tau, min_periods=tau).mean()
    differences = (averages.shift(-tau) - averages).dropna()
    return differences.to_numpy() ** 2


def fragment_windows(fragments: pd.DataFrame) -> pd.DataFrame:
    descriptions = [
        f"fragment {row.fragment}, {row.start:%Y-%m-%d %H:%M:%S}, {row.duration_min:.1f} min,"
        f" zmiana {row.change_kg:+.0f} kg ({row.change_kg_h:+.0f} kg/h), zakłócone próbki {row.spike_samples}"
        for row in fragments.itertuples()
    ]
    return pd.DataFrame({"start": fragments["start"], "end": fragments["end"], "description": descriptions})


def spike_windows(episodes: pd.DataFrame, padding: pd.Timedelta) -> pd.DataFrame:
    descriptions = [
        f"{row.start:%Y-%m-%d %H:%M:%S}, {row.samples} próbek, odchyłka {row.extreme_deviation_kg:+.0f} kg, {row.place}"
        for row in episodes.itertuples()
    ]
    return pd.DataFrame(
        {"start": episodes["start"] - padding, "end": episodes["end"] + padding, "description": descriptions}
    )


class FragmentBrowser:
    def __init__(
        self, samples: pd.DataFrame, events: pd.DataFrame, windows: pd.DataFrame, title: str, width: int
    ) -> None:
        self._samples = samples
        self._events = events
        self._windows = windows.reset_index(drop=True)
        self._title = title
        self._width = width
        self._position = 0

        self._window = widgets.Dropdown(options=list(WINDOWS_S), value=DEFAULT_WINDOW_S, description="okno [s]")
        self._padding = widgets.FloatText(
            value=DEFAULT_PADDING_S, description="margines [s]", layout={"width": "180px"}
        )
        self._previous = widgets.Button(description="◀ poprzedni")
        self._next = widgets.Button(description="następny ▶")
        self._number = widgets.BoundedIntText(
            min=1, max=max(len(self._windows), 1), description="nr", layout={"width": "150px"}
        )
        self._info = widgets.HTML()
        self._plot = widgets.Box()

        for control in (self._window, self._padding):
            control.observe(lambda _: self._show(), names="value")
        self._number.observe(lambda change: self._go_to(change["new"] - 1), names="value")
        self._previous.on_click(lambda _: self._go_to(self._position - 1))
        self._next.on_click(lambda _: self._go_to(self._position + 1))

        count = widgets.Label(f"z {len(self._windows)}")
        self.widget = widgets.VBox(
            [
                widgets.HBox([self._window, self._padding]),
                widgets.HBox([self._previous, self._next, self._number, count]),
                self._info,
                self._plot,
            ]
        )
        self._show()

    def _go_to(self, position: int) -> None:
        if not 0 <= position < len(self._windows) or position == self._position:
            return
        self._position = position
        self._number.value = position + 1
        self._show()

    def _show(self) -> None:
        if self._windows.empty:
            self._info.value = "<b>brak fragmentów</b>"
            return

        window = self._windows.iloc[self._position]
        padding = pd.Timedelta(seconds=self._padding.value)
        start, end = window["start"] - padding, window["end"] + padding
        shown = self._samples[self._samples["timestamp"].between(start, end)]

        self._info.value = f"<b>{self._position + 1}/{len(self._windows)}</b> | {window['description']}"
        for previous in self._plot.children:
            previous.close()
        self._plot.children = [self._figure(shown, start, end)]

    def _figure(self, shown: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> FigureWidgetResampler:
        window_s = self._window.value
        figure = FigureWidgetResampler(
            make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.5, 0.25, 0.25],
                subplot_titles=(
                    "Wskazania [kg]",
                    f"Nachylenie prostej w oknie {window_s} s [kg/h]",
                    f"Odchylenie od prostej w oknie {window_s} s [kg]",
                ),
            )
        )

        time = shown["timestamp"]
        spikes = shown[shown["spike"]]
        traces = [
            (
                go.Scatter(name="wskazanie", mode="lines+markers", marker={"size": 3}, line={"width": 1}),
                time,
                shown["value"],
                1,
            ),
            (
                go.Scatter(
                    name=f"mediana {SPIKE_WINDOW_SAMPLES} próbek", mode="lines", line={"width": 1, "dash": "dot"}
                ),
                time,
                shown["median_kg"],
                1,
            ),
            (
                go.Scatter(name="zakłócenie", mode="markers", marker={"size": 7, "color": "red"}),
                spikes["timestamp"],
                spikes["value"],
                1,
            ),
            (go.Scatter(name="nachylenie", mode="lines", line={"width": 1}), time, shown[slope_column(window_s)], 2),
            (go.Scatter(name="odchylenie", mode="lines", line={"width": 1}), time, shown[std_column(window_s)], 3),
        ]
        for trace, x, y, row in traces:
            figure.add_trace(trace, hf_x=x, hf_y=y, row=row, col=1)

        visible = self._events[(self._events["end"] >= start) & (self._events["start"] <= end)]
        figure.layout.shapes = [_event_band(event.start, event.end, event.rejected) for event in visible.itertuples()]

        normal = shown.loc[~shown["spike"], "value"]
        if not normal.empty:
            margin = max(float(normal.max() - normal.min()) * 0.05, 20.0)
            figure.update_yaxes(range=[normal.min() - margin, normal.max() + margin], row=1, col=1)

        figure.update_xaxes(range=[start, end])
        figure.update_layout(
            title=self._title,
            height=BROWSER_HEIGHT,
            width=self._width,
            autosize=False,
            hovermode="x unified",
            margin={"l": 60, "r": 20, "t": 80, "b": 40},
        )
        return figure


def _event_band(start: pd.Timestamp, end: pd.Timestamp, rejected: bool) -> dict[str, object]:
    return {
        "type": "rect",
        "xref": "x",
        "yref": "paper",
        "x0": start,
        "x1": end,
        "y0": 0,
        "y1": 1,
        "fillcolor": REJECTED_COLOR if rejected else EVENT_COLOR,
        "line": {"width": 0},
        "layer": "below",
    }


def _busy(timestamps: pd.Series, events: pd.DataFrame) -> np.ndarray:
    intervals = events.sort_values("start")
    starts = (intervals["start"] - MARGIN).to_numpy()
    ends = np.maximum.accumulate((intervals["end"] + MARGIN).to_numpy())
    moments = timestamps.to_numpy()

    previous = np.searchsorted(starts, moments, side="right") - 1
    busy = np.zeros(len(moments), dtype=bool)
    started = previous >= 0
    busy[started] = moments[started] <= ends[previous[started]]
    return busy


def _fragments(samples: pd.DataFrame) -> pd.Series:
    free = ~samples["busy"]
    gap = samples["timestamp"].diff() > MAX_SAMPLE_GAP
    run = ((free != free.shift()) | gap).cumsum()

    groups = samples["timestamp"].groupby(run)
    duration = groups.transform("last") - groups.transform("first")
    kept = free & (duration >= MIN_DURATION)

    numbers = run.where(kept).rank(method="dense") - 1
    return numbers.fillna(NO_FRAGMENT).astype(int)


def _rolling_line(samples: pd.DataFrame, window_s: int) -> tuple[pd.Series, pd.Series]:
    inside = samples[samples["fragment"] != NO_FRAGMENT]
    slopes = []
    stds = []

    for _, fragment in inside.groupby("fragment"):
        valid = ~fragment["spike"]
        seconds = (fragment["timestamp"] - fragment["timestamp"].iloc[0]).dt.total_seconds().where(valid)
        values = (fragment["value"] - fragment["value"].iloc[0]).where(valid)

        def mean(series: pd.Series) -> pd.Series:
            return series.rolling(window_s, center=True, min_periods=window_s).mean()

        mean_t, mean_x = mean(seconds), mean(values)
        variance_t = mean(seconds * seconds) - mean_t**2
        covariance = mean(seconds * values) - mean_t * mean_x
        variance_x = mean(values * values) - mean_x**2

        slope = covariance / variance_t
        residual = (variance_x - covariance * slope).clip(lower=0)
        slopes.append(slope * SECONDS_PER_HOUR)
        stds.append(np.sqrt(residual))

    index = samples.index
    return pd.concat(slopes).reindex(index), pd.concat(stds).reindex(index)


def _place(episode: pd.DataFrame) -> str:
    if (episode["fragment"] != NO_FRAGMENT).any():
        return IN_FRAGMENT
    if episode["busy"].any():
        return NEAR_EVENT
    return OUTSIDE
