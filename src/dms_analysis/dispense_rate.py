from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dms_analysis.event_stats import METHODS

DISPENSE = "odsyp"
FIGURE_HEIGHT = 500


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
        figure.add_trace(_points(group, "dispense_rate_kg_s"), row=1, col=column)

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
        figure.add_trace(_points(group, "dispensed_kg"), row=1, col=column)
        figure.add_trace(line, row=1, col=column)

    figure.update_xaxes(title_text="czas trwania [s]")
    figure.update_yaxes(title_text="odsypana masa [kg]", col=1)
    figure.update_layout(showlegend=True)
    return figure


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


def _points(group: pd.DataFrame, column: str) -> go.Scatter:
    return go.Scatter(
        x=group["duration_s"],
        y=group[column],
        mode="markers",
        marker={"size": 8, "opacity": 0.7},
        text=group["start"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        hovertemplate="%{text}<br>%{x:.1f} s<br>%{y:.2f}<extra></extra>",
        showlegend=False,
    )
