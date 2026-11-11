from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.io.formats.style import Styler
from plotly.subplots import make_subplots

from dms_analysis import style
from dms_analysis.data import check
from dms_analysis.data.catalog import Catalog
from dms_analysis.events import store, synthetic
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
GROUPS = (WITH_TEMPERATURES, WITHOUT_TEMPERATURES)
METHOD_COLORS = dict(zip(METHODS, style.CATEGORICAL, strict=True))
PART_COLORS = {synthetic.DAY: style.YELLOW, synthetic.NIGHT: style.VIOLET}
PERIOD_COLORS = (style.BLUE, style.ORANGE, style.AQUA, style.YELLOW)
DELAY_DASH = {0.0: "solid", 2.0: "dash", 3.0: "dot"}
CONVERGENCE_STEPS = (100, 250, 500, 1000)
MEAN_ERROR_RANGE_KG = 10.0
REFILL_MARGIN = pd.Timedelta(minutes=5)
WORST_COUNT = 6
WORST_MARGIN_S = 60.0
TIME_NAMES = {"real": "rzeczywisty", "estimated": "szacowany", "unknown": "zegara"}


def load_all(catalog: Catalog) -> dict[str, Results]:
    return {period.full_id: store.load(period.full_id) for period in stage_periods(catalog)}


def warnings(results: dict[str, Results]) -> list[str]:
    return [warning for item in results.values() if (warning := store.staleness(item)) is not None]


def period_name(full_id: str, item: Results) -> str:
    if item.run["time"] == "estimated":
        return f"{full_id} (czas szacowany ±2 h)"
    return full_id


def _time_dash(item: Results) -> str:
    return "dash" if item.run["time"] == "estimated" else "solid"


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
        for hour in (synthetic.DAY_START_H, synthetic.DAY_END_H):
            figure.add_vline(x=hour, line={"color": style.RED, "width": 1.5, "dash": "dash"}, row=1, col=column)
    figure.update_xaxes(range=[0, 24], dtick=2, title_text="godzina (czas lokalny)")
    figure.update_yaxes(title_text="doby", row=1, col=1)
    figure.update_layout(
        barmode="stack",
        height=380,
        title=(
            f"Godziny pierwszego i ostatniego odsypu doby (wszystkie przyjęte odsypy) · "
            f"obecne granice dnia: {synthetic.DAY_START_H}–{synthetic.DAY_END_H} (czerwone linie)"
        ),
    )
    return figure


def _hour(moments: pd.Series) -> pd.Series:
    return moments.dt.hour + moments.dt.minute / 60 + moments.dt.second / 3600


def coverage_figure(full_id: str, item: Results) -> go.Figure:
    trials = base(item.trials)
    hour = _hour(trials["start"])
    figure = go.Figure()
    for part, color in PART_COLORS.items():
        chosen = trials["part"] == part
        figure.add_trace(
            go.Scatter(
                x=hour[chosen],
                y=trials.loc[chosen, "date"],
                mode="markers",
                marker={"color": color, "size": 4, "opacity": 0.35},
                name=part,
                hoverinfo="skip",
            )
        )
    dates = sorted(trials["date"].unique())
    figure.update_xaxes(range=[0, 24], dtick=2, title_text="godzina startu sztucznego odsypu")
    figure.update_yaxes(type="category", categoryorder="array", categoryarray=dates, autorange="reversed", title_text="doba")
    figure.update_layout(
        title=f"{period_name(full_id, item)} · pokrycie dób próbami (jeden punkt = jedna próba; luki = zdarzenia z marginesem 60 s)",
        height=max(260, 34 * len(dates) + 140),
    )
    return figure


def all_trials(results: dict[str, Results]) -> pd.DataFrame:
    return pd.concat(
        [
            item.trials.assign(period=full_id, period_name=period_name(full_id, item), group=group(item))
            for full_id, item in results.items()
        ],
        ignore_index=True,
    )


def _p95(errors: pd.Series) -> float:
    return float(errors.abs().quantile(0.95))


def convergence_figure(results: dict[str, Results]) -> go.Figure:
    trials = base(all_trials(results))
    measures = (("odchylenie std [kg]", "std"), ("95. percentyl |błędu| [kg]", _p95), ("średni błąd [kg]", "mean"))
    figure = make_subplots(
        rows=len(GROUPS), cols=len(measures), subplot_titles=[f"{g}: {m}" for g in GROUPS for m, _ in measures]
    )
    for row, group_name in enumerate(GROUPS, start=1):
        for column, (_, function) in enumerate(measures, start=1):
            for part, color in PART_COLORS.items():
                chosen = trials[(trials["group"] == group_name) & (trials["part"] == part)]
                points = [chosen[chosen["trial"] <= steps]["error_kg"].agg(function) for steps in CONVERGENCE_STEPS]
                figure.add_trace(
                    go.Scatter(
                        x=list(CONVERGENCE_STEPS),
                        y=points,
                        mode="lines+markers",
                        line={"color": color, "width": 2},
                        marker={"size": 8},
                        name=part,
                        legendgroup=part,
                        showlegend=row == 1 and column == 1,
                        hovertemplate="%{x} prób na dobę i porę: %{y:.1f} kg<extra></extra>",
                    ),
                    row=row,
                    col=column,
                )
    figure.update_xaxes(type="log", tickvals=list(CONVERGENCE_STEPS), title_text="próby na dobę i porę")
    for row in range(1, len(GROUPS) + 1):
        figure.update_yaxes(rangemode="tozero", row=row, col=1)
        figure.update_yaxes(rangemode="tozero", row=row, col=2)
        figure.update_yaxes(range=[-MEAN_ERROR_RANGE_KG, MEAN_ERROR_RANGE_KG], row=row, col=3)
    figure.update_layout(
        title=f"Zbieżność miar błędu z liczbą prób — wariant podstawowy ({variant_label(BASE_METHOD, BASE_DELAY_S)})",
        height=620,
    )
    return figure


def error_histograms(results: dict[str, Results], group_name: str) -> go.Figure:
    trials = all_trials(results)
    trials = trials[trials["group"] == group_name]
    figure = _variant_grid(f"Błąd w teście półsyntetycznym — okresy {group_name} (dzień i noc osobno, każda pora = 100%)")
    for row, method in enumerate(METHODS, start=1):
        for column, delay_s in enumerate(DELAYS_S, start=1):
            chosen = trials[(trials["method"] == method) & (trials["delay_s"] == delay_s)]
            for part, color in PART_COLORS.items():
                errors = chosen[chosen["part"] == part]["error_kg"]
                figure.add_trace(
                    go.Histogram(
                        x=errors.clip(-200, 200),
                        xbins={"start": -210, "end": 210, "size": 20},
                        histnorm="percent",
                        marker={"color": color, "line": {"color": style.SURFACE, "width": 1}},
                        opacity=0.6,
                        name=part,
                        legendgroup=part,
                        showlegend=row == 1 and column == 1,
                        hovertemplate=f"{part}: %{{x}} kg<extra></extra>",
                    ),
                    row=row,
                    col=column,
                )
    figure.update_layout(
        barmode="overlay",
        title=figure.layout.title.text + "<br><sup>błąd = wynik − znana masa [kg], przedziały co 20 kg, obcięty do ±200 kg</sup>",
    )
    figure.update_xaxes(title_text="błąd [kg]", showticklabels=True)
    figure.update_yaxes(title_text="% prób", col=1)
    return figure


def error_table(results: dict[str, Results]) -> Styler:
    trials = all_trials(results)
    grouped = trials.groupby(["group", "part", "method", "delay_s"])["error_kg"]
    table = pd.DataFrame(
        {"średni błąd": grouped.mean(), "odchylenie std": grouped.std(), "95. percentyl |błędu|": grouped.agg(_p95)}
    ).reset_index()
    table["metoda"] = pd.Categorical(table["method"].map(METHOD_NAMES), [METHOD_NAMES[method] for method in METHODS])
    table["d"] = table["delay_s"].map(delay_label)
    wide = table.pivot_table(
        index=["group", "part", "metoda"],
        columns="d",
        values=["odchylenie std", "95. percentyl |błędu|", "średni błąd"],
        observed=True,
    )
    wide = wide.reindex(columns=["odchylenie std", "95. percentyl |błędu|", "średni błąd"], level=0)
    wide.index.names = ["grupa", "pora", "metoda"]
    wide.columns.names = ["miara [kg]", "przerwa"]

    styled = wide.style.format(precision=1)
    for measure in ("odchylenie std", "95. percentyl |błędu|"):
        low, high = float(wide[measure].to_numpy().min()), float(wide[measure].to_numpy().max())
        styled = styled.map(lambda value, low=low, high=high: style.sequential_css(value, low, high), subset=[measure])
    return styled.map(lambda value: style.diverging_css(value, 5), subset=["średni błąd"]).set_caption(
        "Miary błędu testu półsyntetycznego: im ciemniejszy niebieski, tym większy rozrzut; średni błąd: "
        "czerwony = metoda zawyża masę, niebieski = zaniża"
    )


def error_by_hour(results: dict[str, Results]) -> go.Figure:
    trials = base(all_trials(results))
    figure = go.Figure()
    for index, full_id in enumerate(results):
        chosen = trials[trials["period"] == full_id]
        by_hour = chosen.groupby("hour")["error_kg"].std()
        figure.add_trace(
            go.Scatter(
                x=by_hour.index,
                y=by_hour.to_numpy(),
                mode="lines+markers",
                line={"color": PERIOD_COLORS[index], "width": 2, "dash": _time_dash(results[full_id])},
                marker={"size": 8},
                name=period_name(full_id, results[full_id]),
                hovertemplate="godz. %{x}: %{y:.1f} kg<extra>" + period_name(full_id, results[full_id]) + "</extra>",
            )
        )
    figure.add_vrect(x0=-0.5, x1=synthetic.DAY_START_H - 0.5, fillcolor="rgba(98, 80, 214, 0.07)", line_width=0)
    figure.add_vrect(x0=synthetic.DAY_END_H - 0.5, x1=23.5, fillcolor="rgba(98, 80, 214, 0.07)", line_width=0)
    figure.update_xaxes(range=[-0.5, 23.5], dtick=1, title_text="godzina startu (czas lokalny; noc zacieniowana)")
    figure.update_yaxes(title_text="odchylenie std błędu [kg]", rangemode="tozero")
    figure.update_layout(
        title=f"Błąd wg godziny doby — wariant podstawowy ({variant_label(BASE_METHOD, BASE_DELAY_S)})",
        height=450,
    )
    return figure


def error_by_day(results: dict[str, Results]) -> go.Figure:
    trials = base(all_trials(results))
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=list(synthetic.DAY_PARTS), vertical_spacing=0.12)
    for row, part in enumerate(synthetic.DAY_PARTS, start=1):
        chosen = trials[trials["part"] == part]
        by_day = chosen.groupby(["date", "full_day", "period_name"])["error_kg"].std().reset_index()
        for full_day, color, label in ((1, style.BLUE, "doba pełna"), (0, style.EXCLUDED, "doba niepełna")):
            days = by_day[by_day["full_day"] == full_day]
            figure.add_trace(
                go.Bar(
                    x=days["date"],
                    y=days["error_kg"],
                    marker={"color": color},
                    name=label,
                    legendgroup=label,
                    showlegend=row == 1,
                    customdata=days["period_name"],
                    hovertemplate="%{x}: %{y:.1f} kg<br>%{customdata}<extra></extra>",
                ),
                row=row,
                col=1,
            )
        figure.update_yaxes(title_text="odch. std [kg]", row=row, col=1)
    figure.update_xaxes(type="category", categoryorder="category ascending", tickangle=-60)
    figure.update_layout(
        margin={"b": 110},
        title=f"Błąd wg doby — wariant podstawowy ({variant_label(BASE_METHOD, BASE_DELAY_S)})",
        height=620,
        bargap=0.25,
    )
    return figure


def worst_trials(results: dict[str, Results], group_name: str) -> go.Figure:
    trials = base(all_trials(results))
    trials = trials[trials["group"] == group_name]
    worst = trials.reindex(trials["error_kg"].abs().sort_values(ascending=False).index).head(WORST_COUNT)

    titles = [
        f"{row.period}<br>{row.start:%Y-%m-%d %H:%M} · masa {row.mass_kg:.0f} kg · błąd {row.error_kg:+.0f} kg"
        for row in worst.itertuples()
    ]
    figure = make_subplots(rows=WORST_COUNT // 2, cols=2, subplot_titles=titles, vertical_spacing=0.12)
    periods: dict[str, check.PreparedPeriod] = {}
    for index, row in enumerate(worst.itertuples()):
        period = periods.setdefault(row.period, check.load_period(row.period))
        division_kg = float(results[row.period].run["division_kg"])
        _trial_panel(figure, period, row, division_kg, (index // 2 + 1, index % 2 + 1), index == 0)
    figure.update_layout(
        title=f"Największe błędy — okresy {group_name}, wariant podstawowy (szary: oryginał, niebieski: ze sztucznym odsypem)",
        height=330 * (WORST_COUNT // 2) + 120,
    )
    figure.update_annotations(font_size=11)
    return figure


def _trial_panel(
    figure: go.Figure,
    period: check.PreparedPeriod,
    trial: Any,
    division_kg: float,
    position: tuple[int, int],
    legend: bool,
) -> None:
    plot_row, plot_column = position
    start = trial.start
    end = start + pd.Timedelta(seconds=trial.duration_s)
    low, high = start - pd.Timedelta(seconds=WORST_MARGIN_S), end + pd.Timedelta(seconds=WORST_MARGIN_S)
    weight = period.weight[(period.weight["timestamp"] >= low) & (period.weight["timestamp"] <= high)]
    seconds = (weight["timestamp"] - start).dt.total_seconds().to_numpy()
    progress = np.clip(seconds / trial.duration_s, 0.0, 1.0)
    shifted = np.round((weight["value"].to_numpy() - trial.mass_kg * progress) / division_kg) * division_kg

    for values, color, name in ((weight["value"], style.EXCLUDED, "oryginał"), (shifted, style.BLUE, "ze sztucznym odsypem")):
        figure.add_trace(
            go.Scatter(
                x=_naive(weight["timestamp"]),
                y=values,
                mode="lines",
                line={"color": color, "width": 1.5},
                name=name,
                legendgroup=name,
                showlegend=legend,
            ),
            row=plot_row,
            col=plot_column,
        )
    figure.add_vrect(
        x0=_naive_one(start), x1=_naive_one(end), fillcolor=style.SIGNAL_BAND, line_width=0, row=plot_row, col=plot_column
    )
