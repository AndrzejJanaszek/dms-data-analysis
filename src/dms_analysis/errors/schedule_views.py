import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display
from pandas.io.formats.style import Styler
from plotly.subplots import make_subplots

from dms_analysis import style
from dms_analysis.data import check
from dms_analysis.errors import stats
from dms_analysis.errors.balance import NIGHT_TO_NIGHT
from dms_analysis.errors.replay import DAY_SHARE, DAY_SUM, SINGLE
from dms_analysis.events.values import BASE_DELAY_S, BASE_METHOD, DELAYS_S, METHOD_NAMES, METHODS, WINDOW_S

MEASURE_NAMES = {
    stats.MEAN: "średni",
    stats.STD: "odch. std",
    stats.RMSE: "RMSE",
    stats.MAE: "MAE",
    stats.P50: "p50 |·|",
    stats.P95: "p95 |·|",
    stats.P99: "p99 |·| ⚠",
    stats.MAX: "max |·|",
}
TABLE_MEASURES = (stats.MEAN, stats.STD, stats.MAE, stats.P95, stats.MAX)
INVALID_COLOR = "#d9d8d3"
ENVELOPE_BINS = 3000


def base_variant(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[(frame["method"] == BASE_METHOD) & (frame["delay_s"] == BASE_DELAY_S)]


def days_tables(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedules = tables["schedules"].rename(
        columns={
            "date": "doba",
            "dispenses": "odsypy",
            "in_refill": "w trakcie zasypu",
            "y_kg": "Y = Σ mas odsypów poza zasypem [kg]",
            "periods": "okresy",
        }
    )
    bases = tables["bases"].rename(columns={"date": "doba", "period": "okres"})
    return schedules.drop(columns="schedule").set_index("doba"), bases.drop(columns="base").set_index("doba")


def matrix_figure(tables: dict[str, pd.DataFrame]) -> go.Figure:
    pairs = tables["pairs"]
    schedule_dates = tables["schedules"].set_index("schedule")["date"]
    base_dates = tables["bases"].set_index("base")["date"]
    values = pairs.pivot(index="schedule", columns="base", values="e_kg")
    reasons = pairs.pivot(index="schedule", columns="base", values="reason")
    limit = float(np.nanmax(np.abs(values.to_numpy())))
    labels = values.map(lambda value: "" if np.isnan(value) else f"{value:+.0f}")
    hover = np.where(values.isna(), "nieważna: " + reasons.fillna(""), "E = " + labels)

    figure = go.Figure(
        go.Heatmap(
            z=values.isna().astype(float).to_numpy(),
            x=[base_dates[index] for index in values.columns],
            y=[schedule_dates[index] for index in values.index],
            colorscale=[[0, "rgba(0,0,0,0)"], [1, INVALID_COLOR]],
            showscale=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Heatmap(
            z=values.to_numpy(),
            x=[base_dates[index] for index in values.columns],
            y=[schedule_dates[index] for index in values.index],
            text=labels.to_numpy(),
            texttemplate="%{text}",
            customdata=hover,
            hovertemplate="harmonogram %{y}, doba bazowa %{x}<br>%{customdata}<extra></extra>",
            colorscale=[list(point) for point in style.DIVERGING],
            zmin=-limit,
            zmax=limit,
            colorbar={"title": "E [kg]"},
        )
    )
    figure.update_layout(
        title=(
            f"E = suma błędów odsypów harmonogramu odtworzonego na dobie bazowej [kg] · {_variant()}<br>"
            "<span style='font-size:12px'>czerwony: masa odsypów zawyżona (wskazanie spadało), niebieski: zaniżona; "
            "szare: para nieważna (powód po najechaniu)</span>"
        ),
        xaxis={"title": "doba bazowa (bez zdarzeń)", "type": "category"},
        yaxis={"title": "harmonogram (doba robocza)", "type": "category", "autorange": "reversed"},
        height=560,
    )
    return figure


def invalid_pairs(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pairs = tables["pairs"]
    invalid = pairs[pairs["valid"] == 0]
    return (
        invalid.merge(tables["bases"], on="base")
        .groupby(["date", "period", "reason"])
        .agg(pary=("schedule", "size"), odsypy_blokujące=("blocked", "sum"))
        .reset_index()
        .rename(columns={"date": "doba bazowa", "period": "okres", "reason": "powód"})
    )


def _variant(method: str = BASE_METHOD, delay_s: float = BASE_DELAY_S) -> str:
    return f"{METHOD_NAMES[method]}, d = {delay_s:g} s"


def _cell(row: pd.Series, unit: str, measure: str) -> str:
    digits = 2 if unit == "%" else (1 if measure in (stats.MEAN, stats.STD) else 0)
    if np.isnan(row["low"]):
        return f"{row['value']:.{digits}f}"
    return f"{row['value']:.{digits}f} [{row['low']:.{digits}f}…{row['high']:.{digits}f}]"


def measures_table(tables: dict[str, pd.DataFrame]) -> Styler:
    chosen = base_variant(tables["stats"])
    chosen = chosen[chosen["measure"].isin(TABLE_MEASURES)]
    rows = {}
    for (target, part), group in chosen.groupby(["target", "part"], sort=False):
        unit = "%" if target == DAY_SHARE else "kg"
        name = f"{target}" if not part else f"{target} [kg], {part}"
        values = group.set_index("measure")
        rows[name] = {MEASURE_NAMES[measure]: _cell(values.loc[measure], unit, measure) for measure in TABLE_MEASURES}
        rows[name]["liczba"] = f"{int(values['count'].iloc[0])} ({int(values['pairs'].iloc[0])} par)"
    table = pd.DataFrame(rows).T
    return table.style.set_caption(
        f"Miary z przedziałem 95 % (bootstrap po harmonogramach i dobach bazowych niezależnie, B = 2000) · {_variant()}"
    )


def variants_table(tables: dict[str, pd.DataFrame], target: str = DAY_SUM, measure: str = stats.STD) -> Styler:
    chosen = tables["stats"]
    chosen = chosen[(chosen["target"] == target) & (chosen["measure"] == measure)]
    unit = "%" if target == DAY_SHARE else "kg"
    text = chosen.assign(cell=chosen.apply(lambda row: _cell(row, unit, row["measure"]), axis=1)).pivot_table(
        index="method", columns="delay_s", values="cell", aggfunc="first"
    )
    values = chosen.pivot_table(index="method", columns="delay_s", values="value")
    text, values = text.loc[list(METHODS), list(DELAYS_S)], values.loc[list(METHODS), list(DELAYS_S)]
    text.index = [METHOD_NAMES[method] for method in METHODS]
    text.columns = [f"d = {delay_s:g} s" for delay_s in DELAYS_S]
    limit = float(values.abs().max().max())
    colors = values.map(lambda value: style.sequential_css(abs(value), 0.0, limit))
    colors.index, colors.columns = text.index, text.columns
    return text.style.apply(lambda _: colors, axis=None).set_caption(
        f"{target} · {MEASURE_NAMES[measure]} w 9 wariantach [przedział 95 %]"
    )


def histogram_figure(tables: dict[str, pd.DataFrame]) -> go.Figure:
    pairs = tables["pairs"][tables["pairs"]["valid"] == 1]
    figure = make_subplots(rows=1, cols=2, subplot_titles=("E [kg]", "E / Y [%]"), horizontal_spacing=0.1)
    for column, (name, size) in enumerate((("e_kg", 100), ("e_share", 0.5)), start=1):
        figure.add_trace(
            go.Histogram(
                x=pairs[name],
                xbins={"size": size},
                marker={"color": style.BLUE, "line": {"color": style.SURFACE, "width": 1}},
                showlegend=False,
                hovertemplate="%{x}: %{y} par<extra></extra>",
            ),
            row=1,
            col=column,
        )
        figure.add_vline(x=0, line={"color": style.TEXT_SECONDARY, "width": 1}, row=1, col=column)
    figure.update_yaxes(title_text="pary", col=1)
    figure.update_layout(title=f"Rozkład błędu sumy dnia ({len(pairs)} ważnych par) · {_variant()}", height=420)
    return figure


def ratio_table(tables: dict[str, pd.DataFrame]) -> Styler:
    ratio = tables["ratio"].rename(
        columns={
            "schedule": "harmonogram",
            "dispenses": "n odsypów",
            "pairs": "pary",
            "std_e_kg": "std(E) [kg]",
            "sigma_sqrt_n_kg": "σ·√n [kg]",
            "ratio": "std(E) / (σ·√n)",
        }
    )
    sigma = float(ratio["sigma_kg"].iloc[0])
    return (
        ratio.drop(columns="sigma_kg")
        .set_index("harmonogram")
        .style.format(precision=0, na_rep="")
        .format({"std(E) / (σ·√n)": "{:.2f}"})
        .set_caption(
            f"σ = {sigma:.1f} kg = odch. std błędu odtworzonych odsypów w dzień ({_variant()}). Stosunek ≈ 1: błędy "
            "odsypów jednego dnia sumują się jak niezależne; > 1: kumulują się (np. wspólny dryf); < 1: znoszą się. "
            "Wiersz „wszystkie pary”: std(E/√n) / σ."
        )
    )


def balance_comparison(tables: dict[str, pd.DataFrame], balances: pd.DataFrame) -> pd.DataFrame:
    pairs = tables["pairs"][tables["pairs"]["valid"] == 1].merge(tables["schedules"], on="schedule")
    replayed = pairs.groupby("date")["e_kg"].agg(["mean", "min", "max", "size"])
    night = balances[(balances["variant"] == NIGHT_TO_NIGHT) & (balances["valid"] == 1)]
    night = night.groupby("date")[["r_kg", "group"]].first()
    table = replayed.join(night, how="left")
    return pd.DataFrame(
        {
            "R z bilansu noc–noc [kg]": table["r_kg"],
            "grupa": table["group"],
            "E odtworzone: średnia [kg]": table["mean"].round(0),
            "E: min [kg]": table["min"].round(0),
            "E: max [kg]": table["max"].round(0),
            "pary": table["size"],
        }
    ).rename_axis("doba")


class PairBrowser:
    def __init__(self, tables: dict[str, pd.DataFrame]) -> None:
        self._tables = tables
        self._periods: dict[str, check.PreparedPeriod] = {}
        pairs = tables["pairs"][tables["pairs"]["valid"] == 1].merge(tables["schedules"], on="schedule")
        pairs = pairs.merge(tables["bases"], on="base", suffixes=("", "_base"))
        options = [
            (f"{row.date} → {row.date_base} · E = {row.e_kg:+.0f} kg", (row.schedule, row.base))
            for row in pairs.sort_values("e_kg", key=np.abs, ascending=False).itertuples()
        ]
        self._choice = widgets.Dropdown(options=options, description="para", layout={"width": "520px"})
        self._info = widgets.HTML()
        self._output = widgets.Output()
        self._choice.observe(lambda _: self._show(), names="value")
        self._show()

    def widget(self) -> widgets.VBox:
        return widgets.VBox(
            [widgets.HTML("kolejność: największe |E| najpierw"), self._choice, self._info, self._output]
        )

    def _show(self) -> None:
        schedule, base = self._choice.value
        full_id = self._tables["bases"].set_index("base").loc[base, "period"]
        if full_id not in self._periods:
            self._periods = {full_id: check.load_period(full_id)}
        figure = pair_figure(self._periods[full_id], self._tables, schedule, base)
        self._info.value = _utc_note(self._tables, schedule, base)
        with self._output:
            self._output.clear_output(wait=True)
            figure.show()


def _utc_note(tables: dict[str, pd.DataFrame], schedule: int, base: int) -> str:
    replayed = base_variant(tables["replayed"])
    starts = replayed[(replayed["schedule"] == schedule) & (replayed["base"] == base)]["start"]
    first, last = (pd.Timestamp(value, unit="s", tz="UTC") for value in (starts.min(), starts.max()))
    local = first.tz_convert(check.TIMEZONE), last.tz_convert(check.TIMEZONE)
    return (
        f"odsypy na dobie bazowej: {local[0]:%d.%m %H:%M:%S}–{local[1]:%H:%M:%S} lokalnie "
        f"(UTC {first:%d.%m %H:%M:%S}–{last:%H:%M:%S}); wykres w czasie lokalnym"
    )


def pair_figure(period: check.PreparedPeriod, tables: dict[str, pd.DataFrame], schedule: int, base: int) -> go.Figure:
    replayed = base_variant(tables["replayed"])
    replayed = replayed[(replayed["schedule"] == schedule) & (replayed["base"] == base)].sort_values("start")
    starts = pd.to_datetime(replayed["start"], unit="s", utc=True).dt.tz_convert(check.TIMEZONE)
    low = starts.min() - pd.Timedelta(minutes=30)
    high = starts.max() + pd.Timedelta(minutes=30)
    weight = period.weight[(period.weight["timestamp"] >= low) & (period.weight["timestamp"] <= high)]
    reduced = check.envelope(weight, "value", ENVELOPE_BINS)

    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06)
    figure.add_trace(
        go.Scatter(
            x=reduced["timestamp"].dt.tz_localize(None),
            y=reduced["value"],
            mode="lines",
            line={"width": 1, "color": style.WEIGHT},
            name="wskazanie doby bazowej",
        ),
        row=1,
        col=1,
    )
    for start, duration in zip(starts, replayed["duration_s"], strict=True):
        figure.add_vrect(
            x0=(start - pd.Timedelta(seconds=WINDOW_S)).tz_localize(None),
            x1=(start + pd.Timedelta(seconds=duration + BASE_DELAY_S + WINDOW_S)).tz_localize(None),
            fillcolor="rgba(235, 104, 52, 0.25)",
            line_width=0,
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=starts.dt.tz_localize(None),
            y=replayed["error_kg"].cumsum(),
            mode="lines+markers",
            line={"color": style.ORANGE, "width": 2, "shape": "hv"},
            marker={"size": 5},
            customdata=replayed["error_kg"],
            hovertemplate="%{x|%H:%M:%S}: błąd odsypu %{customdata:+.0f} kg, suma %{y:+.0f} kg<extra></extra>",
            name="suma błędów od początku dnia",
        ),
        row=2,
        col=1,
    )
    figure.add_hline(y=0, line={"color": style.TEXT_SECONDARY, "width": 1}, row=2, col=1)
    dates = (
        tables["schedules"].set_index("schedule").loc[schedule, "date"],
        tables["bases"].set_index("base").loc[base, "date"],
    )
    figure.update_yaxes(title_text="wskazanie [kg]", row=1, col=1)
    figure.update_yaxes(title_text="Σ błędów [kg]", row=2, col=1)
    figure.update_layout(
        title=(
            f"harmonogram {dates[0]} na dobie bazowej {dates[1]} · E = {replayed['error_kg'].sum():+.0f} kg · "
            f"{len(replayed)} odsypów<br><span style='font-size:12px'>pomarańczowe pasma: odsypy przeniesione na tę "
            "samą godzinę (z oknami 3 s); dół: suma błędów narastająco; czas lokalny</span>"
        ),
        height=640,
    )
    return figure


def show_pairs(tables: dict[str, pd.DataFrame]) -> None:
    display(PairBrowser(tables).widget())
