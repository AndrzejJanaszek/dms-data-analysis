import ipywidgets as widgets
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display

from dms_analysis import style
from dms_analysis.data import check
from dms_analysis.data.review import grafana_link
from dms_analysis.events.store import Results
from dms_analysis.events.values import (
    BASE_DELAY_S,
    BASE_METHOD,
    DELAYS_S,
    METHOD_NAMES,
    METHODS,
    NEAREST,
    WINDOW_S,
)
from dms_analysis.events.views import DELAY_DASH, METHOD_COLORS, delay_label

ALL_DAYS = "wszystkie doby"
BY_TIME = "po kolei"
BY_SPREAD = "największa rozbieżność"
BY_NO_DROP = "bez spadku"
BY_MASS = "największa masa"
SORTS = (BY_TIME, BY_SPREAD, BY_NO_DROP, BY_MASS)
DEFAULT_MARGIN_S = 60.0
EXCLUDED_NAMES = {"": "do analizy", "okno": "wypada: okno", "kolizja": "wypada: kolizja z zasypem"}


def event_table(item: Results) -> pd.DataFrame:
    masses = item.values.pivot_table(index="event", columns=["method", "delay_s"], values="mass_kg")
    no_drop = item.values.groupby("event")["no_drop"].any()
    table = item.dispenses.set_index("event")
    return table.assign(
        base_kg=masses[(BASE_METHOD, BASE_DELAY_S)],
        spread_kg=masses.max(axis=1) - masses.min(axis=1),
        any_no_drop=no_drop,
        date=table["start"].dt.date.astype(str),
    ).reset_index()


def ordered(events: pd.DataFrame, sort: str) -> pd.DataFrame:
    if sort == BY_SPREAD:
        return events.sort_values(["spread_kg", "start"], ascending=[False, True])
    if sort == BY_NO_DROP:
        return events[events["any_no_drop"]].sort_values("start")
    if sort == BY_MASS:
        return events.sort_values(["base_kg", "start"], ascending=[False, True])
    return events.sort_values("start")


def masses_html(values: pd.DataFrame) -> str:
    table = values.set_index(["method", "delay_s"])["mass_kg"]
    reference = table[(BASE_METHOD, BASE_DELAY_S)]
    header = "".join(f"<th style='padding:2px 10px'>{delay_label(delay_s)}</th>" for delay_s in DELAYS_S)
    rows = []
    for method in METHODS:
        cells = []
        for delay_s in DELAYS_S:
            mass = table[(method, delay_s)]
            base = (method, delay_s) == (BASE_METHOD, BASE_DELAY_S)
            border = "2px solid #0b0b0b" if base else "1px solid #e6e5e1"
            cells.append(
                f"<td style='padding:2px 10px;text-align:right;background:{_deviation_color(mass - reference)};"
                f"border:{border}'>{mass:.0f}</td>"
            )
        swatch = f"<span style='color:{METHOD_COLORS[method]}'>■</span>"
        rows.append(f"<tr><td style='padding:2px 10px'>{swatch} {METHOD_NAMES[method]}</td>{''.join(cells)}</tr>")
    return (
        "<table style='border-collapse:collapse;font-size:13px'>"
        f"<tr><th style='text-align:left;padding:2px 10px'>masa [kg]</th>{header}</tr>{''.join(rows)}</table>"
        "<div style='font-size:11px;color:#52514e'>ramka = wariant podstawowy; kolor = różnica względem niego "
        "(czerwony: więcej, niebieski: mniej, nasycenie pełne przy ±60 kg)</div>"
    )


def _deviation_color(difference_kg: float) -> str:
    strength = min(abs(difference_kg) / 60.0, 1.0) * 0.45
    if difference_kg > 0:
        return f"rgba(227, 73, 72, {strength:.2f})"
    return f"rgba(42, 120, 214, {strength:.2f})"


def event_figure(period: check.PreparedPeriod, item: Results, event: pd.Series, margin_s: float) -> go.Figure:
    start, end = event["start"], event["end"]
    margin = pd.Timedelta(seconds=margin_s)
    weight = period.weight[(period.weight["timestamp"] >= start - margin) & (period.weight["timestamp"] <= end + margin)]

    figure = go.Figure(
        go.Scatter(
            x=_naive(weight["timestamp"]),
            y=weight["value"],
            mode="lines+markers",
            marker={"size": 4, "color": style.WEIGHT},
            line={"width": 1, "color": style.WEIGHT},
            name="wskazanie",
        )
    )
    _event_bands(figure, period, start - margin, end + margin)
    figure.add_vrect(x0=_naive_one(start), x1=_naive_one(end), line={"color": style.TEXT, "width": 1.5}, fillcolor="rgba(0,0,0,0)")
    _value_marks(figure, weight, item.values[item.values["event"] == event["event"]], start, end)

    figure.update_layout(
        height=520,
        yaxis_title="wskazanie [kg]",
        hovermode="x unified",
        title=(
            f"wskazania ±{margin_s:.0f} s · szare pasma: sygnały zdarzeń, czarna ramka: ten odsyp<br>"
            "kreski: wartości przed i po (ciągła d = 0, kreskowana d = 2, kropkowana d = 3 s) · kropki: najbliższa próbka"
        ),
        title_font_size=12,
    )
    return figure


def _event_bands(figure: go.Figure, period: check.PreparedPeriod, low: pd.Timestamp, high: pd.Timestamp) -> None:
    events = period.events[(period.events["end"] >= low) & (period.events["start"] <= high)]
    for other in events.itertuples():
        color = check.MANUAL_COLOR if other.synthetic else style.SIGNAL_BAND
        if other.rejected:
            color = "rgba(168, 166, 159, 0.08)"
        figure.add_vrect(x0=_naive_one(max(other.start, low)), x1=_naive_one(min(other.end, high)), fillcolor=color, line_width=0)


def _value_marks(figure: go.Figure, weight: pd.DataFrame, values: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> None:
    window = pd.Timedelta(seconds=WINDOW_S)
    for method in METHODS:
        color = METHOD_COLORS[method]
        rows = values[values["method"] == method].set_index("delay_s")
        before = rows["before_kg"].iloc[0]
        if method == NEAREST:
            used = weight[weight["timestamp"] < start]["timestamp"].iloc[-1:]
            _point(figure, used, before, color, METHOD_NAMES[method], True)
        else:
            _segment(figure, start - window, start, before, color, "solid", METHOD_NAMES[method], True)
        for delay_s in DELAYS_S:
            after = rows.loc[delay_s, "after_kg"]
            moment = end + pd.Timedelta(seconds=delay_s)
            if method == NEAREST:
                used = weight[weight["timestamp"] >= moment]["timestamp"].iloc[:1]
                _point(figure, used, after, color, METHOD_NAMES[method], False)
            else:
                _segment(figure, moment, moment + window, after, color, DELAY_DASH[delay_s], METHOD_NAMES[method], False)


def _segment(
    figure: go.Figure, low: pd.Timestamp, high: pd.Timestamp, value: float, color: str, dash: str, name: str, legend: bool
) -> None:
    figure.add_trace(
        go.Scatter(
            x=[_naive_one(low), _naive_one(high)],
            y=[value, value],
            mode="lines",
            line={"color": color, "width": 3, "dash": dash},
            name=name,
            legendgroup=name,
            showlegend=legend,
            hovertemplate=f"{name}: {value:.0f} kg<extra></extra>",
        )
    )


def _point(figure: go.Figure, moments: pd.Series, value: float, color: str, name: str, legend: bool) -> None:
    figure.add_trace(
        go.Scatter(
            x=_naive(moments),
            y=[value] * len(moments),
            mode="markers",
            marker={"color": color, "size": 11, "line": {"color": style.SURFACE, "width": 2}},
            name=name,
            legendgroup=name,
            showlegend=legend,
            hovertemplate=f"{name}: {value:.0f} kg<extra></extra>",
        )
    )


def _naive(moments: pd.Series) -> pd.Series:
    return moments.dt.tz_localize(None)


def _naive_one(moment: pd.Timestamp) -> pd.Timestamp:
    return moment.tz_localize(None)


def describe(event: pd.Series, position: int, count: int, link: str) -> str:
    status = EXCLUDED_NAMES[event["excluded"]]
    color = style.TEXT if event["excluded"] == "" else style.RED
    gap_before = "—" if pd.isna(event["gap_before_s"]) else f"{event['gap_before_s']:.0f} s"
    gap_after = "—" if pd.isna(event["gap_after_s"]) else f"{event['gap_after_s']:.0f} s"
    no_drop = " · <b style='color:#e34948'>bez spadku wskazań w części wariantów</b>" if event["any_no_drop"] else ""
    return (
        f"<b>{position + 1}/{count}</b> · {event['start']:%Y-%m-%d %H:%M:%S} – {event['end']:%H:%M:%S}"
        f" · {event['duration_s']:.1f} s · <b style='color:{color}'>{status}</b>"
        f" · przerwa do poprzedniego zdarzenia {gap_before}, do następnego {gap_after}"
        f" · rozbieżność wariantów {event['spread_kg']:.0f} kg{no_drop}"
        f"<br><a href='{link}' target='_blank'>otwórz w Grafanie (±10 min) ↗</a>"
    )


class EventValuesBrowser:
    def __init__(self, results: dict[str, Results], period: str | None = None) -> None:
        self._results = results
        self._period: check.PreparedPeriod | None = None
        self._all = pd.DataFrame()
        self._events = pd.DataFrame()
        self._position = 0

        periods = list(results)
        self._choice = widgets.Dropdown(options=periods, value=period or periods[0], description="okres", layout={"width": "480px"})
        self._day = widgets.Dropdown(options=[ALL_DAYS], value=ALL_DAYS, description="doba", layout={"width": "260px"})
        self._sort = widgets.ToggleButtons(options=list(SORTS), value=BY_TIME, description="kolejność")
        self._margin = widgets.FloatText(value=DEFAULT_MARGIN_S, description="margines [s]", layout={"width": "180px"})
        self._previous = widgets.Button(description="◀ poprzedni")
        self._next = widgets.Button(description="następny ▶")
        self._number = widgets.BoundedIntText(min=1, max=1, description="nr", layout={"width": "150px"})
        self._count = widgets.Label()
        self._info = widgets.HTML()
        self._masses = widgets.HTML()
        self._output = widgets.Output()

        self._choice.observe(lambda _: self._load(), names="value")
        self._day.observe(lambda _: self._select(), names="value")
        self._sort.observe(lambda _: self._select(), names="value")
        self._margin.observe(lambda _: self._show(), names="value")
        self._number.observe(lambda change: self._go_to(change["new"] - 1), names="value")
        self._previous.on_click(lambda _: self._go_to(self._position - 1))
        self._next.on_click(lambda _: self._go_to(self._position + 1))
        self._load()

    def widget(self) -> widgets.VBox:
        return widgets.VBox(
            [
                widgets.HBox([self._choice, self._day, self._margin]),
                self._sort,
                widgets.HBox([self._previous, self._next, self._number, self._count]),
                self._info,
                widgets.HBox([self._output, self._masses]),
            ]
        )

    def _load(self) -> None:
        item = self._results[self._choice.value]
        self._period = check.load_period(self._choice.value)
        self._all = event_table(item)
        self._day.options = [ALL_DAYS, *sorted(self._all["date"].unique())]
        self._day.value = ALL_DAYS
        self._select()

    def _select(self) -> None:
        events = self._all if self._day.value == ALL_DAYS else self._all[self._all["date"] == self._day.value]
        self._events = ordered(events, self._sort.value).reset_index(drop=True)
        self._count.value = f"z {len(self._events)}"
        self._number.max = max(len(self._events), 1)
        self._position = 0
        self._number.value = 1
        self._show()

    def _go_to(self, position: int) -> None:
        if not 0 <= position < len(self._events) or position == self._position:
            return
        self._position = position
        self._number.value = position + 1
        self._show()

    def _show(self) -> None:
        if self._period is None or self._events.empty:
            self._info.value = "<b>brak odsypów w tym wyborze</b>"
            self._masses.value = ""
            self._output.clear_output()
            return

        item = self._results[self._choice.value]
        event = self._events.iloc[self._position]
        link = grafana_link(self._choice.value, event)
        self._info.value = describe(event, self._position, len(self._events), link)
        self._masses.value = masses_html(item.values[item.values["event"] == event["event"]])

        figure = event_figure(self._period, item, event, self._margin.value)
        figure.update_layout(width=820)
        with self._output:
            self._output.clear_output(wait=True)
            figure.show()


def show(results: dict[str, Results], period: str | None = None) -> None:
    display(EventValuesBrowser(results, period).widget())
