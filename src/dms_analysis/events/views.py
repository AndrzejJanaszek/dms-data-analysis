from typing import Any

import pandas as pd
import plotly.graph_objects as go
from pandas.io.formats.style import Styler
from plotly.subplots import make_subplots

from dms_analysis import style
from dms_analysis.data import check
from dms_analysis.data.catalog import Catalog
from dms_analysis.errors.fragments import DAY_END_H, DAY_START_H
from dms_analysis.events import store
from dms_analysis.events.store import Results
from dms_analysis.events.values import (
    BASE_DELAY_S,
    BASE_METHOD,
    DELAYS_S,
    EXCLUDED_COLLISION,
    EXCLUDED_WINDOW,
    METHOD_NAMES,
    METHODS,
    stage_periods,
)

WITH_TEMPERATURES = "z temperaturami"
WITHOUT_TEMPERATURES = "bez temperatur"
METHOD_COLORS = dict(zip(METHODS, style.CATEGORICAL, strict=True))
DELAY_DASH = {0.0: "solid", 2.0: "dash", 3.0: "dot"}
REFILL_MARGIN = pd.Timedelta(minutes=5)
TIME_NAMES = {"real": "rzeczywisty", "estimated": "szacowany", "unknown": "zegara"}


def load_all(catalog: Catalog) -> dict[str, Results]:
    return {period.full_id: store.load(period.full_id) for period in stage_periods(catalog)}


def warnings(results: dict[str, Results]) -> list[str]:
    return [warning for item in results.values() if (warning := store.staleness(item)) is not None]


def group(item: Results) -> str:
    return WITH_TEMPERATURES if item.run["has_temperatures"] else WITHOUT_TEMPERATURES


def delay_label(delay_s: float) -> str:
    return f"d = {delay_s:g} s"


def variant_label(method: str, delay_s: float) -> str:
    return f"{METHOD_NAMES[method]}, {delay_label(delay_s)}"


def analysed_values(item: Results) -> pd.DataFrame:
    kept = item.dispenses[item.dispenses["excluded"] == ""]
    return item.values.merge(kept[["event", "start", "end", "duration_s"]], on="event")


def base(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[(frame["method"] == BASE_METHOD) & (frame["delay_s"] == BASE_DELAY_S)]


def summary_table(results: dict[str, Results]) -> Styler:
    rows = []
    for full_id, item in results.items():
        excluded = item.dispenses["excluded"]
        no_drop = base(analysed_values(item))["no_drop"].sum()
        rows.append(
            {
                "okres": full_id,
                "czas": TIME_NAMES[item.run["time"]],
                "grupa": group(item),
                "odsypy przyjęte": len(item.dispenses),
                "do analizy": int((excluded == "").sum()),
                "wypadły: okno": int((excluded == EXCLUDED_WINDOW).sum()),
                "wypadły: kolizja z zasypem": int((excluded == EXCLUDED_COLLISION).sum()),
                "bez spadku wskazań": int(no_drop),
                "zasypy": len(item.refills),
            }
        )
    table = pd.DataFrame(rows).set_index("okres")
    return (
        table.style.bar(subset=["do analizy"], color="#cfe0f5", vmin=0)
        .bar(subset=["wypadły: okno", "wypadły: kolizja z zasypem"], color="#f6d3c4", vmin=0)
        .bar(subset=["bez spadku wskazań"], color="#f6d3c4", vmin=0)
        .set_caption("Odsypy w okresach etapu 2 (bez spadku wskazań: wariant podstawowy, wśród odsypów do analizy)")
    )


def mass_histograms(results: dict[str, Results]) -> go.Figure:
    figure = make_subplots(rows=len(results), cols=1, shared_xaxes=True, subplot_titles=list(results), vertical_spacing=0.1)
    for row, (full_id, item) in enumerate(results.items(), start=1):
        masses = base(analysed_values(item))["mass_kg"]
        figure.add_trace(
            go.Histogram(
                x=masses,
                xbins={"start": -300, "end": 900, "size": 20},
                marker={"color": style.BLUE, "line": {"color": style.SURFACE, "width": 1}},
                name=full_id,
                showlegend=False,
                hovertemplate="%{x} kg: %{y} odsypów<extra></extra>",
            ),
            row=row,
            col=1,
        )
        figure.update_yaxes(title_text="odsypy", row=row, col=1)
    figure.update_xaxes(title_text="masa odsypu [kg] (przedziały co 20 kg = działka)", showticklabels=True)
    figure.update_layout(
        title=f"Masy odsypów do analizy — wariant podstawowy ({variant_label(BASE_METHOD, BASE_DELAY_S)})",
        height=270 * len(results) + 120,
    )
    return figure


def variant_differences(results: dict[str, Results]) -> go.Figure:
    values = pd.concat([analysed_values(item).assign(period=full_id) for full_id, item in results.items()])
    reference = base(values)[["period", "event", "mass_kg"]].rename(columns={"mass_kg": "base_kg"})
    values = values.merge(reference, on=["period", "event"])
    values["difference_kg"] = values["mass_kg"] - values["base_kg"]

    figure = _variant_grid("Różnica masy względem wariantu podstawowego, wszystkie okresy razem")
    for row, method in enumerate(METHODS, start=1):
        for column, delay_s in enumerate(DELAYS_S, start=1):
            chosen = values[(values["method"] == method) & (values["delay_s"] == delay_s)]["difference_kg"]
            same = (chosen == 0).mean() * 100
            figure.add_trace(
                go.Histogram(
                    x=chosen.clip(-100, 100),
                    xbins={"start": -110, "end": 110, "size": 20},
                    marker={"color": METHOD_COLORS[method], "line": {"color": style.SURFACE, "width": 1}},
                    showlegend=False,
                    hovertemplate="%{x} kg: %{y} odsypów<extra></extra>",
                ),
                row=row,
                col=column,
            )
            _corner_note(figure, row, column, f"bez zmiany: {same:.0f}%")
    figure.update_xaxes(title_text="różnica [kg]", showticklabels=True)
    figure.update_layout(title=figure.layout.title.text + " (przedziały co 20 kg, wartości obcięte do ±100 kg)")
    return figure


def _variant_grid(title: str, shared_y: bool = False) -> go.Figure:
    titles = [variant_label(method, delay_s) for method in METHODS for delay_s in DELAYS_S]
    figure = make_subplots(
        rows=len(METHODS),
        cols=len(DELAYS_S),
        shared_xaxes=True,
        shared_yaxes=shared_y,
        subplot_titles=titles,
        vertical_spacing=0.13,
        horizontal_spacing=0.05,
    )
    figure.update_layout(title=title, height=900)
    figure.update_annotations(font_size=12)
    return figure


def _corner_note(figure: go.Figure, row: int, column: int, text: str) -> None:
    index = (row - 1) * len(DELAYS_S) + column
    suffix = "" if index == 1 else str(index)
    figure.add_annotation(
        text=text,
        xref=f"x{suffix} domain",
        yref=f"y{suffix} domain",
        x=0.98,
        y=0.95,
        showarrow=False,
        xanchor="right",
        font={"size": 11, "color": style.TEXT_SECONDARY},
    )


def no_drop_table(results: dict[str, Results]) -> Styler:
    rows = []
    for full_id, item in results.items():
        values = analysed_values(item)
        counts = values.groupby(["method", "delay_s"])["no_drop"].sum()
        for method in METHODS:
            rows.append(
                {
                    "okres": full_id,
                    "metoda": METHOD_NAMES[method],
                    **{delay_label(delay_s): int(counts[(method, delay_s)]) for delay_s in DELAYS_S},
                }
            )
    table = pd.DataFrame(rows).set_index(["okres", "metoda"])
    high = max(float(table.to_numpy().max()), 1.0)
    return table.style.map(lambda value: style.sequential_css(value, 0, high, style.ORANGE)).set_caption(
        "Liczba odsypów do analizy bez spadku wskazań (masa ≤ 0) wg wariantu"
    )


def refills_table(results: dict[str, Results]) -> Styler:
    frames = []
    for full_id, item in results.items():
        refills = item.refills
        frames.append(
            pd.DataFrame(
                {
                    "okres": full_id,
                    "początek": refills["start"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "czas trwania [min]": (refills["duration_s"] / 60).round(1),
                    "masa: pojedyncze próbki [kg]": refills["mass_nearest_kg"],
                    "masa: mediana 3 s [kg]": refills["mass_median_kg"],
                    "różnica [kg]": refills["difference_kg"],
                    "odsypy w trakcie": refills["dispenses_inside"],
                }
            )
        )
    table = pd.concat(frames, ignore_index=True)
    return (
        table.style.format(precision=0)
        .map(lambda value: style.diverging_css(value, 100), subset=["różnica [kg]"])
        .bar(subset=["odsypy w trakcie"], color="#f6d3c4", vmin=0)
        .hide(axis="index")
        .set_caption("Zasypy (wszystkie wpisane ręcznie): dwa sposoby liczenia masy")
    )


def refill_figures(full_id: str, item: Results) -> list[go.Figure]:
    period = check.load_period(full_id)
    dispenses = item.dispenses
    figures = []
    for refill in item.refills.itertuples():
        start, end = refill.start - REFILL_MARGIN, refill.end + REFILL_MARGIN
        weight = period.weight[(period.weight["timestamp"] >= start) & (period.weight["timestamp"] <= end)]
        figure = go.Figure(
            go.Scatter(
                x=_naive(weight["timestamp"]),
                y=weight["value"],
                mode="lines",
                line={"color": style.WEIGHT, "width": 1.5},
                name="wskazanie",
            )
        )
        figure.add_vrect(x0=_naive_one(refill.start), x1=_naive_one(refill.end), fillcolor=style.WINDOW_BAND, line_width=0)
        inside = dispenses[(dispenses["start"] < end) & (dispenses["end"] > start)]
        for dispense in inside.itertuples():
            figure.add_vrect(
                x0=_naive_one(dispense.start), x1=_naive_one(dispense.end), fillcolor=style.SIGNAL_BAND, line_width=0
            )
        for moment in (refill.start, refill.end):
            figure.add_vline(x=_naive_one(moment), line={"color": style.BLUE, "width": 1, "dash": "dot"})
        figure.update_layout(
            title=(
                f"{full_id} · zasyp {refill.start:%Y-%m-%d %H:%M}<br><sup>mediana 3 s: {refill.mass_median_kg:+,.0f} kg · "
                f"pojedyncze próbki: {refill.mass_nearest_kg:+,.0f} kg · odsypy w trakcie: {refill.dispenses_inside}</sup>"
            ).replace(",", " "),
            height=330,
            showlegend=False,
            yaxis_title="wskazanie [kg]",
        )
        figures.append(figure)
    return figures


def _naive(moments: pd.Series) -> pd.Series:
    return moments.dt.tz_localize(None)


def _naive_one(moment: pd.Timestamp) -> pd.Timestamp:
    return moment.tz_localize(None)


def first_last_hours(results: dict[str, Results]) -> go.Figure:
    rows = []
    for item in results.values():
        dispenses = item.dispenses
        days = dispenses.groupby(dispenses["start"].dt.date)["start"]
        time = "czas rzeczywisty" if item.run["time"] == "real" else "czas szacowany (±2 h)"
        rows.append(pd.DataFrame({"which": "pierwszy odsyp doby", "hour": _hour(days.min()), "time": time}))
        rows.append(pd.DataFrame({"which": "ostatni odsyp doby", "hour": _hour(days.max()), "time": time}))
    hours = pd.concat(rows, ignore_index=True)

    figure = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=["pierwszy odsyp doby", "ostatni odsyp doby"])
    for column, which in enumerate(("pierwszy odsyp doby", "ostatni odsyp doby"), start=1):
        for time, color in (("czas rzeczywisty", style.BLUE), ("czas szacowany (±2 h)", style.EXCLUDED)):
            chosen = hours[(hours["which"] == which) & (hours["time"] == time)]
            figure.add_trace(
                go.Histogram(
                    x=chosen["hour"],
                    xbins={"start": 0, "end": 24, "size": 1},
                    marker={"color": color, "line": {"color": style.SURFACE, "width": 1}},
                    name=time,
                    legendgroup=time,
                    showlegend=column == 1,
                    hovertemplate="godz. %{x}: %{y} dób<extra></extra>",
                ),
                row=1,
                col=column,
            )
        for hour in (DAY_START_H, DAY_END_H):
            figure.add_vline(x=hour, line={"color": style.RED, "width": 1.5, "dash": "dash"}, row=1, col=column)
    figure.update_xaxes(range=[0, 24], dtick=2, title_text="godzina (czas lokalny)")
    figure.update_yaxes(title_text="doby", row=1, col=1)
    figure.update_layout(
        barmode="stack",
        height=380,
        title=(
            f"Godziny pierwszego i ostatniego odsypu doby (wszystkie przyjęte odsypy) · "
            f"obecne granice dnia: {DAY_START_H}–{DAY_END_H} (czerwone linie)"
        ),
    )
    return figure


def _hour(moments: pd.Series) -> pd.Series:
    return moments.dt.hour + moments.dt.minute / 60 + moments.dt.second / 3600
