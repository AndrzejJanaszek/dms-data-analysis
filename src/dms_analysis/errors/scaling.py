from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.io.formats.style import Styler
from plotly.subplots import make_subplots

from dms_analysis import style
from dms_analysis.errors import stats
from dms_analysis.errors.fragments import DAY, DAY_PARTS, NIGHT
from dms_analysis.errors.summary import REAL_TIME_SCOPE
from dms_analysis.events import store as events_store
from dms_analysis.events.values import BASE_DELAY_S, BASE_METHOD, base_values

PART_COLORS = {DAY: style.YELLOW, NIGHT: style.VIOLET}
RELATIVE_MEASURES = (stats.STD, stats.P95, stats.P99)
SCALING_MEASURES = (stats.STD, stats.P95)
REFERENCE_SLOPES = {0.0: "nachylenie 0: sam szum", 0.5: "0,5: dryf losowy (∝ √T)", 1.0: "1: stały dryf (∝ T)"}
MEASURE_NAMES = {stats.STD: "odch. std", stats.P95: "p95 |e|", stats.P99: "p99 |e| ⚠"}


@dataclass(frozen=True)
class DispenseRate:
    median_kg_s: float
    low_kg_s: float
    high_kg_s: float
    dispenses: int


def dispense_rate(periods: list[str]) -> DispenseRate:
    frames = []
    for full_id in periods:
        results = events_store.load(full_id)
        values = base_values(results.dispenses, results.values)
        frames.append(values[(values["excluded"] == "") & (values["mass_kg"] > 0)])
    dispenses = pd.concat(frames, ignore_index=True)
    rate = dispenses["mass_kg"] / dispenses["duration_s"]
    return DispenseRate(
        median_kg_s=float(rate.median()),
        low_kg_s=float(rate.quantile(0.1)),
        high_kg_s=float(rate.quantile(0.9)),
        dispenses=len(dispenses),
    )


def base_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[
        (summary["scope"] == REAL_TIME_SCOPE)
        & (summary["method"] == BASE_METHOD)
        & (summary["delay_s"] == BASE_DELAY_S)
    ]


def relative_table(summary: pd.DataFrame, rate: DispenseRate) -> Styler:
    chosen = base_summary(summary)
    chosen = chosen[chosen["measure"].isin(RELATIVE_MEASURES)]
    mass = rate.median_kg_s * chosen["duration_s"]
    relative = chosen.assign(
        value=100 * chosen["value_kg"] / mass,
        low=100 * chosen["low_kg"] / mass,
        high=100 * chosen["high_kg"] / mass,
    )
    cells = relative.assign(cell=relative.apply(_percent_cell, axis=1))
    text = cells.pivot_table(index="duration_s", columns=["part", "measure"], values="cell", aggfunc="first")
    values = cells.pivot_table(index="duration_s", columns=["part", "measure"], values="value")
    order = [(part, measure) for part in DAY_PARTS for measure in RELATIVE_MEASURES]
    text, values = text[order], values[order]

    text.columns = pd.MultiIndex.from_tuples([(part, MEASURE_NAMES[measure]) for part, measure in order])
    text.index = [f"{duration:.0f} s · {rate.median_kg_s * duration:.0f} kg" for duration in text.index]
    text.index.name = "T · typowa masa"
    limit = float(values.max().max())
    colors = values.map(lambda value: style.sequential_css(value, 0.0, limit))
    colors.index, colors.columns = text.index, text.columns
    return text.style.apply(lambda _: colors, axis=None).set_caption(
        f"Błąd względny = błąd / typowa masa odsypu o czasie T (szybkość {rate.median_kg_s:.1f} kg/s = mediana "
        f"{rate.dispenses} odsypów etapu 2 z okresów z czasem rzeczywistym; 80 % odsypów: "
        f"{rate.low_kg_s:.1f}–{rate.high_kg_s:.1f} kg/s) · [przedział 95 %, szybkość traktowana jako stała]"
    )


def _percent_cell(row: pd.Series) -> str:
    return f"{row['value']:.1f} % [{row['low']:.1f}…{row['high']:.1f}]"


def slopes(summary: pd.DataFrame) -> pd.DataFrame:
    chosen = base_summary(summary).sort_values("duration_s")
    rows = []
    for (part, measure), group in chosen.groupby(["part", "measure"]):
        if measure not in SCALING_MEASURES:
            continue
        durations = group["duration_s"].to_numpy(dtype=float)
        values = group["value_kg"].to_numpy(dtype=float)
        local = np.diff(np.log(values)) / np.diff(np.log(durations))
        whole = np.log(values[-1] / values[0]) / np.log(durations[-1] / durations[0])
        rows.append(
            {
                "pora": part,
                "miara": MEASURE_NAMES[measure],
                **{f"{durations[i]:.0f}→{durations[i + 1]:.0f} s": round(local[i], 2) for i in range(len(local))},
                f"całość {durations[0]:.0f}→{durations[-1]:.0f} s": round(whole, 2),
            }
        )
    return pd.DataFrame(rows).set_index(["pora", "miara"])


def scaling_figure(summary: pd.DataFrame) -> go.Figure:
    chosen = base_summary(summary)
    figure = make_subplots(
        rows=1,
        cols=len(SCALING_MEASURES),
        subplot_titles=[f"{MEASURE_NAMES[measure]} [kg]" for measure in SCALING_MEASURES],
        horizontal_spacing=0.08,
    )
    for column, measure in enumerate(SCALING_MEASURES, start=1):
        for part in DAY_PARTS:
            rows = chosen[(chosen["measure"] == measure) & (chosen["part"] == part)].sort_values("duration_s")
            figure.add_trace(
                go.Scatter(
                    x=rows["duration_s"],
                    y=rows["value_kg"],
                    mode="lines+markers",
                    line={"color": PART_COLORS[part], "width": 2},
                    marker={"size": 7},
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": rows["high_kg"] - rows["value_kg"],
                        "arrayminus": rows["value_kg"] - rows["low_kg"],
                        "color": PART_COLORS[part],
                    },
                    name=part,
                    legendgroup=part,
                    showlegend=column == 1,
                    hovertemplate=f"{part}, T = %{{x}} s: %{{y:.1f}} kg<extra></extra>",
                ),
                row=1,
                col=column,
            )
            _reference_lines(figure, rows, column, show_legend=column == 1 and part == DAY)
    figure.update_xaxes(type="log", title_text="czas trwania T [s] (skala log.)", tickvals=[10, 20, 30, 45, 60, 90])
    figure.update_yaxes(type="log", title_text="błąd [kg] (skala log.)", col=1)
    figure.update_yaxes(type="log", col=2)
    figure.update_layout(
        title=(
            "Jak błąd rośnie z czasem trwania · okresy z czasem rzeczywistym · mediana, d = 3 s · pionowe kreski: przedział 95 %<br>"
            "<span style='font-size:12px'>szare linie: nachylenia odniesienia 0 / 0,5 / 1, zaczepione w T = 10 s dnia</span>"
        ),
        height=520,
    )
    return figure


def _reference_lines(figure: go.Figure, rows: pd.DataFrame, column: int, show_legend: bool) -> None:
    if rows.empty or rows["part"].iloc[0] != DAY:
        return
    start, end = float(rows["duration_s"].iloc[0]), float(rows["duration_s"].iloc[-1])
    anchor = float(rows["value_kg"].iloc[0])
    for slope, name in REFERENCE_SLOPES.items():
        figure.add_trace(
            go.Scatter(
                x=[start, end],
                y=[anchor, anchor * (end / start) ** slope],
                mode="lines",
                line={"color": style.EXCLUDED, "width": 1, "dash": {0.0: "dot", 0.5: "dash", 1.0: "solid"}[slope]},
                name=name,
                legendgroup=f"slope-{slope}",
                showlegend=show_legend,
                hoverinfo="skip",
            ),
            row=1,
            col=column,
        )
