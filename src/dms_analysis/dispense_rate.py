from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dms_analysis.event_stats import METHODS, NEAREST

DISPENSE = "odsyp"
REFILL = "zasyp"
FIGURE_HEIGHT = 500
KG_PER_T = 1000
COLORS = plotly.colors.qualitative.Plotly
TARGETS = {
    "dispense_rate_kg_s": "średnia prędkość [kg/s]",
    "duration_s": "czas trwania [s]",
    "dispensed_kg": "odsypana masa [kg]",
}


@dataclass(frozen=True)
class XAxis:
    title: str
    unit: str


X_AXES = {
    "content_t": XAxis("zawartość przed odsypem [t]", "t"),
    "hours_since_refill": XAxis("czas od końca ostatniego pełnego zasypu [h]", "h"),
    "hour_of_day": XAxis("godzina dnia (zegar RPi z przesunięciem) [h]", "h"),
}


@dataclass(frozen=True)
class DispenseSelection:
    kept: pd.DataFrame
    rejected: pd.DataFrame


def dispenses(table: pd.DataFrame) -> pd.DataFrame:
    selected = table[(table["role"] == DISPENSE) & ~table["collision"]]
    return selected.assign(
        dispensed_kg=-selected["difference_kg"],
        dispense_rate_kg_s=-selected["rate_kg_s"],
    )


def without_non_negative(table: pd.DataFrame) -> DispenseSelection:
    return _split(table, table["difference_kg"] >= 0)


def trimmed(table: pd.DataFrame, lower_percent: float, upper_percent: float) -> DispenseSelection:
    by_method = table.groupby("method")["difference_kg"]
    low = by_method.transform(lambda values: values.quantile(lower_percent / 100))
    high = by_method.transform(lambda values: values.quantile(1 - upper_percent / 100))
    return _split(table, (table["difference_kg"] < low) | (table["difference_kg"] > high))


def correlations(table: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for name, group in _by_method(table):
        duration = group["duration_s"]
        rate = group["dispense_rate_kg_s"]
        slope, intercept = np.polyfit(duration, group["dispensed_kg"], 1)
        rows.append(
            {
                "method": name,
                "n": len(group),
                "pearson": duration.corr(rate),
                "spearman": duration.rank().corr(rate.rank()),
                "slope_kg_s": slope,
                "intercept_kg": intercept,
                "time_offset_s": -intercept / slope,
            }
        )

    return pd.DataFrame(rows).set_index("method")


def trim_sensitivity(table: pd.DataFrame, levels: list[float]) -> pd.DataFrame:
    frames = []

    for lower in levels:
        for upper in levels:
            result = correlations(trimmed(table, lower, upper).kept)
            frames.append(result.assign(lower_percent=lower, upper_percent=upper))

    return pd.concat(frames).reset_index().set_index(["lower_percent", "upper_percent", "method"])


def rate_figure(table: pd.DataFrame, title: str, width: int) -> go.Figure:
    figure = _method_columns(title, width)

    for column, (_, group) in enumerate(_by_method(table), start=1):
        figure.add_trace(_points(group, group["duration_s"], "dispense_rate_kg_s", "s"), row=1, col=column)

    figure.update_xaxes(title_text="czas trwania [s]")
    figure.update_yaxes(title_text="średnia prędkość [kg/s]", col=1)
    return figure


def mass_figure(table: pd.DataFrame, title: str, width: int) -> go.Figure:
    figure = _method_columns(title, width)
    fits = correlations(table)

    for column, (name, group) in enumerate(_by_method(table), start=1):
        fit = fits.loc[name]
        line_x = np.array([min(0.0, fit["time_offset_s"]), group["duration_s"].max()])
        line = go.Scatter(
            x=line_x,
            y=fit["slope_kg_s"] * line_x + fit["intercept_kg"],
            mode="lines",
            line={"width": 2, "color": "black"},
            name=f"{fit['slope_kg_s']:.2f} kg/s, przesunięcie {fit['time_offset_s']:.1f} s",
        )
        figure.add_trace(_points(group, group["duration_s"], "dispensed_kg", "s"), row=1, col=column)
        figure.add_trace(line, row=1, col=column)

    figure.update_xaxes(title_text="czas trwania [s]")
    figure.update_yaxes(title_text="odsypana masa [kg]", col=1)
    figure.update_layout(showlegend=True)
    return figure


def full_refills(table: pd.DataFrame, min_kg: float) -> pd.DataFrame:
    refills = table[(table["role"] == REFILL) & (table["method"] == NEAREST) & (table["difference_kg"] >= min_kg)]
    return refills[["start", "end", "duration_s", "difference_kg"]].sort_values("start").reset_index(drop=True)


def with_context(table: pd.DataFrame, refills: pd.DataFrame, clock_offset_h: float) -> pd.DataFrame:
    labels = [f"od {bound:%Y-%m-%d %H:%M}" for bound in [table["start"].min(), *refills["start"]]]
    index = refills["start"].searchsorted(table["start"])
    last_end = refills["end"].reindex(index - 1).set_axis(table.index)
    local = table["start"] + pd.Timedelta(hours=clock_offset_h)

    return table.assign(
        period=[labels[position] for position in index],
        content_t=table["before_kg"] / KG_PER_T,
        hours_since_refill=(table["start"] - last_end).dt.total_seconds() / 3600,
        hour_of_day=local.dt.hour + local.dt.minute / 60,
    )


def x_correlations(table: pd.DataFrame, x: str) -> pd.DataFrame:
    rows = []
    complete = table.dropna(subset=[x])

    for column, label in TARGETS.items():
        for name, group in _by_method(complete):
            values = group[column]
            rows.append(
                {
                    "variable": label,
                    "method": name,
                    "n": len(group),
                    "pearson": group[x].corr(values),
                    "spearman": group[x].rank().corr(values.rank()),
                    f"slope_per_{X_AXES[x].unit}": np.polyfit(group[x], values, 1)[0],
                }
            )

    return pd.DataFrame(rows).set_index(["variable", "method"])


def scatter_figure(
    table: pd.DataFrame, x: str, y: str, title: str, width: int, color: str | None = None, trend: bool = True
) -> go.Figure:
    figure = _method_columns(title, width)
    complete = table.dropna(subset=[x])
    labels = [None] if color is None else sorted(complete[color].unique())

    for column_index, (_, method_group) in enumerate(_by_method(complete), start=1):
        for label_index, label in enumerate(labels):
            group = method_group if label is None else method_group[method_group[color] == label]
            points = _points(group, group[x], y, X_AXES[x].unit)

            if label is not None:
                points.update(
                    name=str(label),
                    legendgroup=str(label),
                    showlegend=column_index == 1,
                    marker_color=COLORS[label_index % len(COLORS)],
                )

            figure.add_trace(points, row=1, col=column_index)

        if trend:
            figure.add_trace(_trend(method_group, x, y), row=1, col=column_index)

    figure.update_xaxes(title_text=X_AXES[x].title)
    figure.update_yaxes(title_text=TARGETS[y], col=1)
    figure.update_layout(showlegend=color is not None)
    return figure


def box_figure(table: pd.DataFrame, x: str, y: str, bin_size: float, title: str, width: int) -> go.Figure:
    figure = _method_columns(title, width)
    complete = table.dropna(subset=[x])

    for column_index, (_, group) in enumerate(_by_method(complete), start=1):
        bins = np.floor(group[x] / bin_size) * bin_size
        labels = [f"{low:g}–{low + bin_size:g}" for low in bins]
        order = [f"{low:g}–{low + bin_size:g}" for low in sorted(bins.unique())]
        box = go.Box(x=labels, y=group[y], boxpoints="all", jitter=0.4, pointpos=0, showlegend=False)
        figure.add_trace(box, row=1, col=column_index)
        figure.update_xaxes(categoryorder="array", categoryarray=order, row=1, col=column_index)

    figure.update_xaxes(title_text=X_AXES[x].title)
    figure.update_yaxes(title_text=TARGETS[y], col=1)
    return figure


def _trend(group: pd.DataFrame, x: str, y: str) -> go.Scatter:
    slope, intercept = np.polyfit(group[x], group[y], 1)
    line_x = np.array([group[x].min(), group[x].max()])
    return go.Scatter(
        x=line_x,
        y=slope * line_x + intercept,
        mode="lines",
        line={"width": 2, "color": "black"},
        name=f"trend {slope:.3f} na {X_AXES[x].unit}",
        showlegend=False,
        hoverinfo="name",
    )


def _split(table: pd.DataFrame, rejected_rows: pd.Series) -> DispenseSelection:
    rejected_events = table.loc[rejected_rows, "start"].unique()
    rejected = table["start"].isin(rejected_events)
    return DispenseSelection(kept=table[~rejected], rejected=table[rejected])


def _by_method(table: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [(name, table[table["method"] == name]) for name in METHODS]


def _method_columns(title: str, width: int) -> go.Figure:
    figure = make_subplots(rows=1, cols=len(METHODS), shared_yaxes=True, subplot_titles=list(METHODS))
    figure.update_layout(title=title, width=width, height=FIGURE_HEIGHT, showlegend=False)
    return figure


def _points(group: pd.DataFrame, x: pd.Series, column: str, x_unit: str) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=group[column],
        mode="markers",
        marker={"size": 8, "opacity": 0.7},
        text=group["start"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        hovertemplate=f"%{{text}}<br>%{{x:.1f}} {x_unit}<br>%{{y:.2f}}<extra></extra>",
        showlegend=False,
    )
