import ipywidgets as widgets
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display

from dms_analysis import style
from dms_analysis.data import check
from dms_analysis.data.review import grafana_link
from dms_analysis.errors.fragments import DAY_PARTS
from dms_analysis.errors.grid import DURATIONS_S
from dms_analysis.errors.views import variant_label
from dms_analysis.events.values import BASE_DELAY_S, WINDOW_S

DEFAULT_MARGIN_S = 120.0
DEFAULT_DURATION_S = 30


class WorstStarts:
    def __init__(self, results: dict[str, dict[str, pd.DataFrame]]) -> None:
        self._results = results
        self._periods: dict[str, check.PreparedPeriod] = {}
        self._starts = pd.DataFrame()
        self._position = 0

        self._period = widgets.Dropdown(options=list(results), description="okres", layout={"width": "480px"})
        self._duration = widgets.Dropdown(options=list(DURATIONS_S), value=DEFAULT_DURATION_S, description="T [s]")
        self._part = widgets.ToggleButtons(options=list(DAY_PARTS), description="pora")
        self._margin = widgets.FloatText(value=DEFAULT_MARGIN_S, description="margines [s]", layout={"width": "180px"})
        self._previous = widgets.Button(description="◀ poprzedni")
        self._next = widgets.Button(description="następny ▶")
        self._number = widgets.BoundedIntText(min=1, max=1, description="nr", layout={"width": "150px"})
        self._count = widgets.Label()
        self._info = widgets.HTML()
        self._output = widgets.Output()

        for control in (self._period, self._duration, self._part):
            control.observe(lambda _: self._select(), names="value")
        self._margin.observe(lambda _: self._show(), names="value")
        self._number.observe(lambda change: self._go_to(change["new"] - 1), names="value")
        self._previous.on_click(lambda _: self._go_to(self._position - 1))
        self._next.on_click(lambda _: self._go_to(self._position + 1))

        self._select()

    def widget(self) -> widgets.VBox:
        return widgets.VBox(
            [
                widgets.HBox([self._period, self._duration, self._margin]),
                self._part,
                widgets.HBox([self._previous, self._next, self._number, self._count]),
                self._info,
                self._output,
            ]
        )

    def _select(self) -> None:
        worst = self._results[self._period.value]["grid_worst"]
        chosen = worst[(worst["duration_s"] == self._duration.value) & (worst["part"] == self._part.value)]
        self._starts = (
            chosen.assign(size=chosen["error_kg"].abs()).sort_values("size", ascending=False).reset_index(drop=True)
        )
        self._count.value = f"z {len(self._starts)}"
        self._number.max = max(len(self._starts), 1)
        self._position = 0
        self._number.value = 1
        self._show()

    def _go_to(self, position: int) -> None:
        if not 0 <= position < len(self._starts) or position == self._position:
            return
        self._position = position
        self._number.value = position + 1
        self._show()

    def _period_data(self) -> check.PreparedPeriod:
        full_id = self._period.value
        if full_id not in self._periods:
            self._periods = {full_id: check.load_period(full_id)}
        return self._periods[full_id]

    def _show(self) -> None:
        if self._starts.empty:
            self._info.value = "<b>brak startów</b>"
            self._output.clear_output()
            return

        row = self._starts.iloc[self._position]
        start = pd.Timestamp(row["start"], unit="s", tz="UTC").tz_convert(check.TIMEZONE)
        end = start + pd.Timedelta(seconds=float(row["duration_s"]))
        link = grafana_link(self._period.value, pd.Series({"start": start, "end": end}))
        self._info.value = (
            f"<b>{self._position + 1}/{len(self._starts)}</b> · start {start:%Y-%m-%d %H:%M:%S} lokalnie "
            f"(UTC {start.tz_convert('UTC'):%H:%M:%S}) · "
            f"T = {row['duration_s']:.0f} s · <b>błąd {row['error_kg']:+.0f} kg</b> ({variant_label()}) · "
            f"<a href='{link}' target='_blank'>otwórz w Grafanie (±10 min) ↗</a>"
        )
        figure = start_figure(self._period_data(), start, end, float(row["error_kg"]), self._margin.value)
        with self._output:
            self._output.clear_output(wait=True)
            figure.show()


def start_figure(
    period: check.PreparedPeriod, start: pd.Timestamp, end: pd.Timestamp, error_kg: float, margin_s: float
) -> go.Figure:
    margin = pd.Timedelta(seconds=margin_s)
    weight = period.weight[
        (period.weight["timestamp"] >= start - margin) & (period.weight["timestamp"] <= end + margin)
    ]
    window = pd.Timedelta(seconds=WINDOW_S)
    after_start = end + pd.Timedelta(seconds=BASE_DELAY_S)

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
    figure.add_vrect(x0=_naive_one(start), x1=_naive_one(end), fillcolor=style.SIGNAL_BAND, line_width=0)
    for low, high in ((start - window, start), (after_start, after_start + window)):
        figure.add_vrect(x0=_naive_one(low), x1=_naive_one(high), fillcolor=style.WINDOW_BAND, line_width=0)
    _median_segment(figure, weight, start - window, start, "przed")
    _median_segment(figure, weight, after_start, after_start + window, "po")
    _other_events(figure, period, start - margin, end + margin)

    figure.update_layout(
        height=480,
        yaxis_title="wskazanie [kg]",
        hovermode="x unified",
        title=(
            f"udawany odsyp bez masy: błąd {error_kg:+.0f} kg · szare pasmo: czas trwania T, "
            "niebieskie: okna 3 s (przed, po z d = 3 s), czerwone kreski: mediany okien<br>"
            "zielone pasma: zdarzenia w pobliżu (prawdziwe co najmniej 60 s od okien), szare: odrzucone przez reguły"
        ),
        title_font_size=12,
    )
    return figure


def _median_segment(figure: go.Figure, weight: pd.DataFrame, low: pd.Timestamp, high: pd.Timestamp, name: str) -> None:
    samples = weight[(weight["timestamp"] >= low) & (weight["timestamp"] < high)]["value"]
    value = float(samples.median())
    figure.add_trace(
        go.Scatter(
            x=[_naive_one(low), _naive_one(high)],
            y=[value, value],
            mode="lines",
            line={"color": style.RED, "width": 3},
            name=f"mediana {name}: {value:.0f} kg",
        )
    )


def _other_events(figure: go.Figure, period: check.PreparedPeriod, low: pd.Timestamp, high: pd.Timestamp) -> None:
    events = period.events[(period.events["end"] >= low) & (period.events["start"] <= high)]
    for event in events.itertuples():
        figure.add_vrect(
            x0=_naive_one(max(event.start, low)),
            x1=_naive_one(min(event.end, high)),
            fillcolor="rgba(168, 166, 159, 0.15)" if event.rejected else check.EVENT_COLOR,
            line_width=0,
        )


def _naive(moments: pd.Series) -> pd.Series:
    return moments.dt.tz_localize(None)


def _naive_one(moment: pd.Timestamp) -> pd.Timestamp:
    return moment.tz_localize(None)


def show(results: dict[str, dict[str, pd.DataFrame]]) -> None:
    display(WorstStarts(results).widget())
