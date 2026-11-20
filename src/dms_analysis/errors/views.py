import sqlite3

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.io.formats.style import Styler
from plotly.subplots import make_subplots

from dms_analysis import style
from dms_analysis.data.catalog import Catalog
from dms_analysis.data.paths import prepared_path
from dms_analysis.errors import stats, store
from dms_analysis.errors.fragments import DAY, DAY_END_H, DAY_PARTS, DAY_START_H, NIGHT, part_of_hour
from dms_analysis.errors.grid import DURATIONS_S
from dms_analysis.errors.summary import REAL_TIME_SCOPE
from dms_analysis.events.values import BASE_DELAY_S, BASE_METHOD, DELAYS_S, METHOD_NAMES, METHODS, stage_periods

PART_COLORS = {DAY: style.YELLOW, NIGHT: style.VIOLET}
DURATION_COLORS = dict(
    zip(DURATIONS_S, ("#9ec5f0", "#5f9fe3", style.BLUE, "#1d5aa5", "#123c70", "#0b2344"), strict=True)
)
TIME_NAMES = {"real": "rzeczywisty", "estimated": "szacowany ±2 h", "unknown": "zegara"}
MEASURE_NAMES = {
    stats.MEAN: "średni",
    stats.STD: "odch. std",
    stats.RMSE: "RMSE",
    stats.MAE: "MAE",
    stats.P50: "p50 |e|",
    stats.P95: "p95 |e|",
    stats.P99: "p99 |e| ⚠",
    stats.MAX: "max |e|",
}
TABLE_MEASURES = (stats.MEAN, stats.STD, stats.P95, stats.P99, stats.MAX)
CONTROL_PERIODS = ("swietochlowice-1/0004-2026-10-30", "swietochlowice-1/0005-2026-11-02")
CONTROL_STD_KG = {(10, DAY): 20, (10, NIGHT): 10, (30, DAY): 31, (30, NIGHT): 11, (90, DAY): 62, (90, NIGHT): 13}
HISTOGRAM_LIMIT_KG = 400


def load_all(catalog: Catalog) -> dict[str, dict[str, pd.DataFrame]]:
    return {period.full_id: store.load_period(period.full_id) for period in stage_periods(catalog)}


def warnings(results: dict[str, dict[str, pd.DataFrame]]) -> list[str]:
    found = []
    for full_id, tables in results.items():
        connection = sqlite3.connect(f"file:{prepared_path(full_id)}?mode=ro", uri=True)
        try:
            prepared_at = connection.execute("SELECT prepared_at FROM period").fetchone()[0]
        finally:
            connection.close()
        if prepared_at != tables["run"]["prepared_at"].iloc[0]:
            found.append(f"{full_id}: przygotowana baza nowsza niż wyniki — uruchom ponownie errors grid")
    return found


def real_time(results: dict[str, dict[str, pd.DataFrame]]) -> list[str]:
    return [full_id for full_id, tables in results.items() if tables["run"]["time"].iloc[0] == "real"]


def counts_of(results: dict[str, dict[str, pd.DataFrame]], periods: list[str]) -> pd.DataFrame:
    frames = [results[full_id]["grid_counts"].assign(period=full_id) for full_id in periods]
    counts = pd.concat(frames, ignore_index=True)
    return counts.assign(part=counts["hour"].map(part_of_hour))


def variant(counts: pd.DataFrame, method: str = BASE_METHOD, delay_s: float = BASE_DELAY_S) -> pd.DataFrame:
    return counts[(counts["method"] == method) & (counts["delay_s"] == delay_s)]


def variant_label(method: str = BASE_METHOD, delay_s: float = BASE_DELAY_S) -> str:
    return f"{METHOD_NAMES[method]}, d = {delay_s:g} s"


def inputs_table(results: dict[str, dict[str, pd.DataFrame]]) -> Styler:
    rows = []
    for full_id, tables in results.items():
        run = tables["run"].iloc[0]
        counts = variant(tables["grid_counts"])
        reference = counts[counts["duration_s"] == 30]
        skipped = tables["grid_skipped"]
        days = tables["days"]
        rows.append(
            {
                "okres": full_id,
                "czas": TIME_NAMES[run["time"]],
                "doby": f"{days['date'].iloc[0]} – {days['date'].iloc[-1]}",
                "pełne doby": int(days["full_day"].sum()),
                "starty T = 30 s": int(reference["starts"].sum()),
                "w tym noc": int(reference[reference["hour"].map(part_of_hour) == NIGHT]["starts"].sum()),
                "pominięte T = 30 s (< 3 próbki)": int(skipped[skipped["duration_s"] == 30]["starts"].sum()),
                "godziny ze startami": int(reference.groupby(["date", "hour"]).ngroups),
            }
        )
    table = pd.DataFrame(rows).set_index("okres")
    return (
        table.style.bar(subset=["starty T = 30 s", "w tym noc"], color="#cfe0f5", vmin=0)
        .format(thousands=" ")
        .set_caption("Okresy etapu 3: starty testu na siatce (co 1 s we fragmentach bez zdarzeń), przykład T = 30 s")
    )


def control_table(results: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    counts = variant(counts_of(results, [full_id for full_id in CONTROL_PERIODS if full_id in results]))
    rows = []
    for (duration_s, part), expected in CONTROL_STD_KG.items():
        chosen = counts[(counts["duration_s"] == duration_s) & (counts["part"] == part)]
        rows.append(
            {
                "T [s]": duration_s,
                "pora": part,
                "odch. std teraz [kg]": round(stats.point_measures(chosen)[stats.STD], 1),
                "odch. std z dyskusji 15.11 [kg]": expected,
            }
        )
    return pd.DataFrame(rows).set_index(["T [s]", "pora"])


def _cell(row: pd.Series) -> str:
    digits = 1 if row["measure"] in (stats.MEAN, stats.STD) else 0
    if np.isnan(row["low_kg"]):
        return f"{row['value_kg']:.{digits}f}"
    return f"{row['value_kg']:.{digits}f} [{row['low_kg']:.{digits}f}…{row['high_kg']:.{digits}f}]"


def measures_table(
    summary: pd.DataFrame, scope: str = REAL_TIME_SCOPE, method: str = BASE_METHOD, delay_s: float = BASE_DELAY_S
) -> Styler:
    chosen = summary[
        (summary["scope"] == scope)
        & (summary["method"] == method)
        & (summary["delay_s"] == delay_s)
        & summary["measure"].isin(TABLE_MEASURES)
    ]
    text = chosen.assign(cell=chosen.apply(_cell, axis=1)).pivot_table(
        index="duration_s", columns=["part", "measure"], values="cell", aggfunc="first"
    )
    values = chosen.pivot_table(index="duration_s", columns=["part", "measure"], values="value_kg")
    order = [(part, measure) for part in DAY_PARTS for measure in TABLE_MEASURES]
    text, values = text[order], values[order]
    text.columns = pd.MultiIndex.from_tuples([(part, MEASURE_NAMES[measure]) for part, measure in order])
    text.index.name = "T [s]"
    days = chosen.groupby("part")["days"].first()

    limit = float(values.abs().max().max())
    colors = values.map(lambda value: style.sequential_css(abs(value), 0.0, limit))
    colors.columns = text.columns
    return text.style.apply(lambda _: colors, axis=None).set_caption(
        f"{scope} · {variant_label(method, delay_s)} · wartość [przedział 95 % z bootstrapu po dobach]; "
        f"doby: dzień {days.get(DAY, 0)}, noc {days.get(NIGHT, 0)}; max bez przedziału; p99 ⚠ — ogon, mało dób"
    )


def variants_table(summary: pd.DataFrame, measure: str = stats.STD, duration_s: int = 30) -> Styler:
    chosen = summary[
        (summary["scope"] == REAL_TIME_SCOPE) & (summary["duration_s"] == duration_s) & (summary["measure"] == measure)
    ]
    text = chosen.assign(cell=chosen.apply(_cell, axis=1)).pivot_table(
        index="method", columns=["part", "delay_s"], values="cell", aggfunc="first"
    )
    values = chosen.pivot_table(index="method", columns=["part", "delay_s"], values="value_kg")
    order = [(part, delay_s) for part in DAY_PARTS for delay_s in DELAYS_S]
    text, values = text.loc[list(METHODS), order], values.loc[list(METHODS), order]
    text.index = [METHOD_NAMES[method] for method in METHODS]
    text.columns = pd.MultiIndex.from_tuples([(part, f"d = {delay_s:g} s") for part, delay_s in order])

    limit = float(values.max().max())
    colors = values.map(lambda value: style.sequential_css(value, 0.0, limit))
    colors.index, colors.columns = text.index, text.columns
    return text.style.apply(lambda _: colors, axis=None).set_caption(
        f"{REAL_TIME_SCOPE} · T = {duration_s} s · {MEASURE_NAMES[measure]} [kg] w 9 wariantach"
    )


def periods_table(summary: pd.DataFrame, results: dict[str, dict[str, pd.DataFrame]], duration_s: int = 30) -> Styler:
    chosen = summary[
        (summary["scope"] != REAL_TIME_SCOPE)
        & (summary["method"] == BASE_METHOD)
        & (summary["delay_s"] == BASE_DELAY_S)
        & (summary["duration_s"] == duration_s)
        & summary["measure"].isin((stats.STD, stats.P95))
    ]
    text = chosen.assign(cell=chosen.apply(_cell, axis=1)).pivot_table(
        index="scope", columns=["part", "measure"], values="cell", aggfunc="first"
    )
    values = chosen.pivot_table(index="scope", columns=["part", "measure"], values="value_kg")
    order = [(part, measure) for part in DAY_PARTS for measure in (stats.STD, stats.P95)]
    text, values = text[order], values[order]
    text.columns = pd.MultiIndex.from_tuples([(part, MEASURE_NAMES[measure]) for part, measure in order])

    labels = {
        full_id: f"{full_id} · {TIME_NAMES[tables['run']['time'].iloc[0]]} · "
        f"{tables['days']['date'].iloc[0]} – {tables['days']['date'].iloc[-1]}"
        for full_id, tables in results.items()
    }
    text.index = [labels[full_id] for full_id in text.index]
    limit = float(values.max().max())
    colors = values.map(lambda value: style.sequential_css(value, 0.0, limit))
    colors.index, colors.columns = text.index, text.columns
    return text.style.apply(lambda _: colors, axis=None).set_caption(
        f"Okresy osobno · T = {duration_s} s · {variant_label()} · [przedział 95 % z bootstrapu po dobach okresu]"
    )


def by_hour_figure(results: dict[str, dict[str, pd.DataFrame]]) -> go.Figure:
    counts = variant(counts_of(results, real_time(results)))
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("odchylenie standardowe błędu [kg]", "p95 |błędu| [kg]"),
    )
    for duration_s in DURATIONS_S:
        rows = []
        for hour, group in counts[counts["duration_s"] == duration_s].groupby("hour"):
            rows.append({"hour": hour, **stats.point_measures(group), "starts": int(group["starts"].sum())})
        table = pd.DataFrame(rows)
        for row, measure in enumerate((stats.STD, stats.P95), start=1):
            figure.add_trace(
                go.Scatter(
                    x=table["hour"],
                    y=table[measure],
                    mode="lines+markers",
                    line={"color": DURATION_COLORS[duration_s], "width": 2},
                    marker={"size": 5},
                    name=f"T = {duration_s} s",
                    legendgroup=str(duration_s),
                    showlegend=row == 1,
                    customdata=table["starts"],
                    hovertemplate=f"T = {duration_s} s, godz. %{{x}}: %{{y:.1f}} kg (%{{customdata}} startów)<extra></extra>",
                ),
                row=row,
                col=1,
            )
    _night_bands(figure)
    figure.update_xaxes(title_text="godzina startu (czas lokalny)", dtick=1, row=2, col=1)
    figure.update_layout(title=f"Błąd wg godziny · okresy z czasem rzeczywistym razem · {variant_label()}", height=720)
    return figure


def _night_bands(figure: go.Figure) -> None:
    for low, high in ((-0.5, DAY_START_H - 0.5), (DAY_END_H - 0.5, 23.5)):
        figure.add_vrect(x0=low, x1=high, fillcolor="rgba(98, 80, 214, 0.07)", line_width=0)


def histogram_figure(results: dict[str, dict[str, pd.DataFrame]], duration_s: int = 30) -> go.Figure:
    counts = variant(counts_of(results, real_time(results)))
    counts = counts[counts["duration_s"] == duration_s]
    figure = go.Figure()
    for part in DAY_PARTS:
        chosen = counts[counts["part"] == part].groupby("error_kg")["starts"].sum()
        share = chosen / chosen.sum()
        figure.add_trace(
            go.Bar(
                x=share.index,
                y=share.to_numpy(),
                name=part,
                marker={"color": PART_COLORS[part]},
                opacity=0.75,
                customdata=chosen.to_numpy(),
                hovertemplate=f"{part}: błąd %{{x}} kg — %{{y:.2%}} (%{{customdata}} startów)<extra></extra>",
            )
        )
    figure.update_layout(
        barmode="overlay",
        title=f"Rozkład błędu · T = {duration_s} s · {variant_label()} · okresy z czasem rzeczywistym (oś Y logarytmiczna)",
        xaxis={
            "title": "błąd = wartość przed − wartość po [kg] (dodatni: wskazanie spadło)",
            "range": [-HISTOGRAM_LIMIT_KG, HISTOGRAM_LIMIT_KG],
        },
        yaxis={"title": "udział startów", "type": "log"},
        height=460,
    )
    return figure


def coverage_figure(results: dict[str, dict[str, pd.DataFrame]], duration_s: int = 30) -> go.Figure:
    frames = []
    for full_id, tables in results.items():
        counts = variant(tables["grid_counts"])
        counts = counts[counts["duration_s"] == duration_s]
        frames.append(counts.groupby(["date", "hour"])["starts"].sum().reset_index())
    table = pd.concat(frames).groupby(["date", "hour"])["starts"].sum().unstack("hour").reindex(columns=range(24))
    table = table.reindex(sorted(table.index))

    figure = go.Figure(
        go.Heatmap(
            z=table.to_numpy(),
            x=list(range(24)),
            y=table.index,
            colorscale=[list(point) for point in style.SEQUENTIAL],
            zmin=0,
            zmax=3600,
            colorbar={"title": "startów"},
            hovertemplate="%{y}, godz. %{x}: %{z} startów<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"Pokrycie: liczba startów na godzinę (T = {duration_s} s, wszystkie okresy; puste = brak danych lub same zdarzenia)",
        xaxis={"title": "godzina", "dtick": 1},
        yaxis={"autorange": "reversed", "type": "category"},
        height=max(420, 18 * len(table) + 140),
    )
    return figure
