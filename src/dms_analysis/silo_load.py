import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dms_analysis.between_events_fragments import NO_FRAGMENT
from dms_analysis.synthetic import ALL, DAY, NIGHT, OTHER, part_of_day

SMOOTHING_S = 10
THRESHOLDS_KG = (40, 160)
SPAN_S = 60
MIN_CHUNK_S = 1800
LOAD_BIN_KG = 5000
PARTS = (NIGHT, OTHER, DAY)
PART_COLORS = {NIGHT: "#1f4e9c", OTHER: "#999999", DAY: "#e08a00"}
SECONDS_PER_HOUR = 3600
FIGURE_HEIGHT = 900
TIMELINE_RESAMPLE = "1min"
SWING_COLUMNS = {threshold_kg: f"wahnięć ≥ {threshold_kg} kg na godzinę" for threshold_kg in THRESHOLDS_KG}
AMPLITUDE = f"zmiana w wahnięciu ≥ {THRESHOLDS_KG[0]} kg, mediana [kg]"
RESIDUAL = f"odchylenie od prostej w odcinkach {SPAN_S} s, mediana [kg]"
MEASURES = (*SWING_COLUMNS.values(), AMPLITUDE, RESIDUAL)


def hour_chunks(samples: pd.DataFrame, swing_table: pd.DataFrame, residuals: pd.DataFrame) -> pd.DataFrame:
    inside = samples[(samples["fragment"] != NO_FRAGMENT) & ~samples["spike"]]
    groups = inside.groupby(["fragment", inside["timestamp"].dt.floor("h")])
    chunks = pd.DataFrame(
        {
            "start": groups["timestamp"].first(),
            "seconds": groups.size(),
            "load_kg": groups["value"].median(),
        }
    )
    chunks.index.names = ["fragment", "hour"]

    chosen = swing_table[swing_table["smoothing_s"] == SMOOTHING_S]
    for threshold_kg, column in SWING_COLUMNS.items():
        counts = _by_chunk(chosen[chosen["threshold_kg"] == threshold_kg], "start").size()
        chunks[column] = SECONDS_PER_HOUR * counts.reindex(chunks.index, fill_value=0) / chunks["seconds"]

    smallest = chosen[chosen["threshold_kg"] == THRESHOLDS_KG[0]]
    chunks[AMPLITUDE] = _by_chunk(smallest, "start")["amplitude_kg"].median()

    lines = residuals[(residuals["span_s"] == SPAN_S) & (residuals["degree"] == 1)]
    chunks[RESIDUAL] = _by_chunk(lines, "start")["std_kg"].median()

    chunks = chunks[chunks["seconds"] >= MIN_CHUNK_S].reset_index()
    chunks["part_of_day"] = chunks["hour"].dt.hour.map(part_of_day)
    return chunks


def correlation_table(chunks: pd.DataFrame) -> pd.DataFrame:
    columns = {}
    for part in (*PARTS, ALL):
        chosen = chunks if part == ALL else chunks[chunks["part_of_day"] == part]
        columns[f"{part} ({len(chosen)})"] = [_rank_correlation(chosen["load_kg"], chosen[measure]) for measure in MEASURES]
    return pd.DataFrame(columns, index=list(MEASURES))


def fragment_medians(chunks: pd.DataFrame) -> pd.DataFrame:
    return chunks.groupby(["fragment", "part_of_day"])[["load_kg", *MEASURES]].median().reset_index()


def load_bin_table(chunks: pd.DataFrame, measure: str) -> pd.DataFrame:
    bins = (chunks["load_kg"] // LOAD_BIN_KG * LOAD_BIN_KG).astype(int)
    labels = bins.map(lambda low: f"{low // 1000}–{(low + LOAD_BIN_KG) // 1000} t")
    medians = chunks.groupby([bins, labels, chunks["part_of_day"]])[measure].median().unstack("part_of_day")
    counts = chunks.groupby([bins, labels]).size().rename("godzin")
    table = pd.concat([medians.reindex(columns=list(PARTS)), counts], axis=1)
    return table.droplevel(0).rename_axis("stan silosu")


def load_figure(chunks: pd.DataFrame, width: int) -> go.Figure:
    figure = make_subplots(rows=len(MEASURES), cols=1, shared_xaxes=True, vertical_spacing=0.04)

    for row, measure in enumerate(MEASURES, start=1):
        for part in PARTS:
            chosen = chunks[chunks["part_of_day"] == part]
            figure.add_trace(
                go.Scatter(
                    name=part,
                    legendgroup=part,
                    showlegend=row == 1,
                    x=chosen["load_kg"] / 1000,
                    y=chosen[measure],
                    mode="markers",
                    marker={"size": 6, "color": PART_COLORS[part], "opacity": 0.7},
                    customdata=np.stack([chosen["fragment"], chosen["hour"].dt.strftime("%Y-%m-%d %H:00")], axis=1),
                    hovertemplate="fragment %{customdata[0]}, %{customdata[1]}<br>%{x:.1f} t: %{y:.1f}<extra></extra>",
                ),
                row=row,
                col=1,
            )
        figure.update_yaxes(title_text=measure, title_font={"size": 11}, row=row, col=1)

    figure.update_xaxes(title_text="stan silosu: mediana wskazania w godzinie [t]", row=len(MEASURES), col=1)
    figure.update_layout(
        title="Wahania wskazań a stan silosu (punkt = jedna godzina fragmentu bez zdarzeń)",
        width=width,
        height=FIGURE_HEIGHT,
    )
    return figure


def timeline_figure(samples: pd.DataFrame, chunks: pd.DataFrame, width: int) -> go.Figure:
    readings = samples.set_index("timestamp")["value"].where(~samples["spike"].to_numpy())
    readings = readings.resample(TIMELINE_RESAMPLE).median()
    swings = SWING_COLUMNS[THRESHOLDS_KG[0]]

    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    figure.add_trace(
        go.Scatter(name=f"wskazanie (mediana {TIMELINE_RESAMPLE})", x=readings.index, y=readings, mode="lines"),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            name="stan silosu w godzinie fragmentu",
            x=chunks["start"],
            y=chunks["load_kg"],
            mode="markers",
            marker={"size": 5, "color": "black"},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            name=swings,
            x=chunks["start"],
            y=chunks[swings],
            mode="markers",
            marker={"size": 6, "color": chunks["part_of_day"].map(PART_COLORS)},
        ),
        row=2,
        col=1,
    )

    figure.update_yaxes(title_text="wskazanie [kg]", row=1, col=1)
    figure.update_yaxes(title_text=swings, title_font={"size": 11}, row=2, col=1)
    figure.update_layout(
        title="Stan silosu i liczba wahnięć w czasie (kolor punktów na dole = pora dnia)",
        width=width,
        height=FIGURE_HEIGHT * 2 // 3,
    )
    return figure


def _rank_correlation(first: pd.Series, second: pd.Series) -> float:
    pairs = pd.DataFrame({"first": first, "second": second}).dropna().rank()
    return float(pairs["first"].corr(pairs["second"]))


def _by_chunk(table: pd.DataFrame, time_column: str) -> pd.api.typing.DataFrameGroupBy:
    return table.groupby(["fragment", table[time_column].dt.floor("h").rename("hour")])
