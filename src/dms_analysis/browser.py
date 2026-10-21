from collections.abc import Callable
from pathlib import Path
from typing import Any

import ipywidgets as widgets
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dms_analysis.compare import CATEGORIES, UNCHANGED, compare_events
from dms_analysis.prepared import load_session, load_sessions, load_sources, load_variants
from dms_analysis.raw import SessionData
from dms_analysis.signals import PIN_ROLES, SignalEvents, cleaned_changes

DEFAULT_OFFSET_SECONDS = 5.0
FIGURE_HEIGHT = 700
FIGURE_WIDTH = 1200
ROW_SPACING = 1.5
BAND_MARGIN = 0.2
EVENT_COLOR = "rgba(0, 160, 0, 0.15)"
REJECTED_COLOR = "rgba(220, 0, 0, 0.15)"
GAP_COLOR = "rgba(255, 140, 0, 0.25)"
LAYER_EVENT_COLOR = "rgba(0, 160, 0, 0.35)"
LAYER_REJECTED_COLOR = "rgba(220, 0, 0, 0.35)"
WEIGHT_MARGIN_SHARE = 0.05
MIN_WEIGHT_MARGIN_KG = 20
ALL_ROLES = "wszystkie"
ALL_DIFFERENCES = "wszystkie różnice"
ALL_CATEGORIES = "wszystkie"
ACCEPTED = "przyjęte"
REJECTED = "odrzucone"
ALL_STATUSES = "wszystkie"
DEFAULT_SOURCE = "sqlite-2026-10-18"
DEFAULT_SESSION = 9
MAX_DURATION_SECONDS = 100_000.0

Selection = Callable[[pd.DataFrame, str, float, float], pd.DataFrame]


def all_events(signal_events: SignalEvents) -> pd.DataFrame:
    events = pd.concat(
        [signal_events.events.assign(rejected=False), signal_events.too_short.assign(rejected=True)],
        ignore_index=True,
    )
    return events.sort_values(["pin", "start"]).reset_index(drop=True)


def events_by_duration(events: pd.DataFrame, role: str, low: float, high: float) -> pd.DataFrame:
    seconds = events["duration"].dt.total_seconds()
    selected = events[_has_role(events, role) & (seconds >= low) & (seconds < high)]

    return pd.DataFrame(
        {
            "pin": selected["pin"],
            "start": selected["start"],
            "end": selected["end"],
            "marks": [[(row.start, row.end, _event_color(row.rejected), None)] for row in selected.itertuples()],
            "description": [_event_description(row) for row in selected.itertuples()],
        }
    ).sort_values("start", ignore_index=True)


def events_by_gap(events: pd.DataFrame, role: str, low: float, high: float) -> pd.DataFrame:
    previous = events.groupby("pin").shift().assign(pin=events["pin"])
    gap = (events["start"] - previous["end"]).dt.total_seconds()
    selected = _has_role(events, role) & (gap >= low) & (gap < high)
    after = events[selected]
    before = previous[selected]

    marks = [
        [
            (first.start, first.end, _event_color(first.rejected), None),
            (first.end, second.start, GAP_COLOR, None),
            (second.start, second.end, _event_color(second.rejected), None),
        ]
        for first, second in zip(before.itertuples(), after.itertuples(), strict=True)
    ]
    descriptions = [
        f"przerwa {seconds:.3f} s | przed: {_event_description(first)} | po: {_event_description(second)}"
        for seconds, first, second in zip(gap[selected], before.itertuples(), after.itertuples(), strict=True)
    ]

    return pd.DataFrame(
        {
            "pin": after["pin"],
            "start": before["start"],
            "end": after["end"],
            "marks": marks,
            "description": descriptions,
        }
    ).sort_values("start", ignore_index=True)


class EventBrowser:
    def __init__(
        self,
        data: SessionData,
        signal_events: SignalEvents,
        select: Selection,
        low: float,
        high: float,
        title: str,
        width: int = FIGURE_WIDTH,
    ) -> None:
        self._events = all_events(signal_events)
        self._select_windows = select
        self._viewer = _Viewer(data, [("po obróbce", cleaned_changes(signal_events.events, data.pins))], title, width)

        self._role = widgets.Dropdown(options=[*PIN_ROLES.values(), ALL_ROLES], description="rola")
        self._low = widgets.FloatText(value=low, description="od [s]", layout={"width": "160px"})
        self._high = widgets.FloatText(value=high, description="do [s]", layout={"width": "160px"})

        for control in (self._role, self._low, self._high):
            control.observe(lambda _: self._select(), names="value")

        self.widget = self._viewer.widget([self._role, self._low, self._high])
        self._select()

    def _select(self) -> None:
        windows = self._select_windows(self._events, self._role.value, self._low.value, self._high.value)
        self._viewer.set_windows(windows)


class DiffBrowser:
    def __init__(
        self,
        data: SessionData,
        events_a: SignalEvents,
        events_b: SignalEvents,
        names: tuple[str, str],
        title: str,
        width: int = FIGURE_WIDTH,
    ) -> None:
        self._names = names
        self.clusters = compare_events(all_events(events_a), all_events(events_b))
        layers = [
            (f"A: {names[0]}", cleaned_changes(events_a.events, data.pins)),
            (f"B: {names[1]}", cleaned_changes(events_b.events, data.pins)),
        ]
        self._viewer = _Viewer(data, layers, title, width)

        self._role = widgets.Dropdown(options=[*PIN_ROLES.values(), ALL_ROLES], description="rola")
        self._category = widgets.Dropdown(
            options=[ALL_DIFFERENCES, *CATEGORIES, ALL_CATEGORIES], description="kategoria"
        )

        for control in (self._role, self._category):
            control.observe(lambda _: self._select(), names="value")

        self.widget = self._viewer.widget([self._role, self._category])
        self._select()

    def _select(self) -> None:
        clusters = self.clusters[_has_role(self.clusters, self._role.value)]
        category = self._category.value
        if category == ALL_DIFFERENCES:
            clusters = clusters[clusters["category"] != UNCHANGED]
        elif category != ALL_CATEGORIES:
            clusters = clusters[clusters["category"] == category]

        self._viewer.set_windows(
            pd.DataFrame(
                {
                    "pin": clusters["pin"],
                    "start": clusters["start"],
                    "end": clusters["end"],
                    "marks": [_layer_marks(row.a, 0) + _layer_marks(row.b, 1) for row in clusters.itertuples()],
                    "description": [self._diff_description(row) for row in clusters.itertuples()],
                }
            )
        )

    def _diff_description(self, cluster: Any) -> str:
        return (
            f"<b>{cluster.category}</b>, pin {cluster.pin} {cluster.role}"
            f"<br>A ({self._names[0]}): {_versions_description(cluster.a)}"
            f"<br>B ({self._names[1]}): {_versions_description(cluster.b)}"
        )


class PreparedBrowser:
    def __init__(self, db_path: Path, title: str = "Przygotowane dane", width: int = FIGURE_WIDTH) -> None:
        self._db_path = db_path
        self._title = title
        self._events = pd.DataFrame()
        self._loading = False
        empty = pd.DataFrame()
        self._viewer = _Viewer(SessionData(empty, empty, empty), [("po obróbce", empty)], title, width)

        sources = load_sources(db_path)
        variants = load_variants(db_path)
        self._source = widgets.Dropdown(
            options=list(zip(sources["source_name"], sources["source"], strict=True)), description="baza"
        )
        if DEFAULT_SOURCE in set(sources["source"]):
            self._source.value = DEFAULT_SOURCE
        self._session = widgets.Dropdown(description="sesja")
        self._variant = widgets.Dropdown(
            options=[(f"{row.variant} – {row.description}", row.variant) for row in variants.itertuples()],
            description="wariant",
            layout={"width": "520px"},
        )
        self._role = widgets.Dropdown(options=[*PIN_ROLES.values(), ALL_ROLES], description="rola")
        self._status = widgets.Dropdown(options=[ALL_STATUSES, ACCEPTED, REJECTED], description="status")
        self._low = widgets.FloatText(value=0, description="od [s]", layout={"width": "160px"})
        self._high = widgets.FloatText(value=MAX_DURATION_SECONDS, description="do [s]", layout={"width": "200px"})

        self._source.observe(lambda _: self._source_changed(), names="value")
        for control in (self._session, self._variant):
            control.observe(lambda _: self._load(), names="value")
        for control in (self._role, self._status, self._low, self._high):
            control.observe(lambda _: self._select(), names="value")

        self.widget = widgets.VBox(
            [
                widgets.HBox([self._source, self._session, self._variant]),
                self._viewer.widget([self._role, self._status, self._low, self._high]),
            ]
        )
        self._source_changed(DEFAULT_SESSION)

    def _source_changed(self, session: int | None = None) -> None:
        sessions = load_sessions(self._db_path, self._source.value)
        options = [
            (f"sesja {row.session_id} ({row.first_at:%Y-%m-%d %H:%M} – {row.last_at:%Y-%m-%d %H:%M})", row.session_id)
            for row in sessions.itertuples()
        ]
        self._loading = True
        self._session.options = options
        ids = [value for _, value in options]
        self._session.value = session if session in ids else (ids[-1] if ids else None)
        self._loading = False
        self._load()

    def _load(self) -> None:
        if self._loading or self._session.value is None:
            return
        prepared = load_session(self._db_path, self._source.value, int(self._session.value), self._variant.value)
        self._events = prepared.events
        self._viewer.set_source(prepared.data, [("po obróbce", prepared.cleaned)])
        self._select()

    def _select(self) -> None:
        events = self._events
        if self._status.value != ALL_STATUSES:
            events = events[events["rejected"] == (self._status.value == REJECTED)]
        self._viewer.set_windows(events_by_duration(events, self._role.value, self._low.value, self._high.value))


class _Viewer:
    def __init__(self, data: SessionData, layers: list[tuple[str, pd.DataFrame]], title: str, width: int) -> None:
        self._data = data
        self._layers = layers
        self._windows = pd.DataFrame()
        self._position = 0

        self._offset = widgets.FloatText(
            value=DEFAULT_OFFSET_SECONDS, description="margines [s]", layout={"width": "180px"}
        )
        self._previous = widgets.Button(description="◀ poprzednie")
        self._next = widgets.Button(description="następne ▶")
        self._number = widgets.BoundedIntText(min=1, max=1, description="nr", layout={"width": "150px"})
        self._count = widgets.Label()
        self._info = widgets.HTML()
        self._figure = _empty_figure([name for name, _ in layers], title, width)

        self._offset.observe(lambda _: self._show(), names="value")
        self._number.observe(lambda change: self._go_to(change["new"] - 1), names="value")
        self._previous.on_click(lambda _: self._go_to(self._position - 1))
        self._next.on_click(lambda _: self._go_to(self._position + 1))

    def widget(self, filters: list[widgets.Widget]) -> widgets.VBox:
        return widgets.VBox(
            [
                widgets.HBox([*filters, self._offset]),
                widgets.HBox([self._previous, self._next, self._number, self._count]),
                self._info,
                self._figure,
            ]
        )

    def set_source(self, data: SessionData, layers: list[tuple[str, pd.DataFrame]]) -> None:
        self._data = data
        self._layers = layers

    def set_windows(self, windows: pd.DataFrame) -> None:
        self._windows = windows.reset_index(drop=True)
        self._count.value = f"z {len(self._windows)}"
        self._number.max = max(len(self._windows), 1)
        self._position = 0
        self._number.value = 1
        self._show()

    def _go_to(self, position: int) -> None:
        if not 0 <= position < len(self._windows) or position == self._position:
            return
        self._position = position
        self._number.value = position + 1
        self._show()

    def _show(self) -> None:
        if self._windows.empty:
            self._info.value = "<b>brak zdarzeń w zadanym przedziale</b>"
            _clear(self._figure)
            return

        window = self._windows.iloc[self._position]
        offset = pd.Timedelta(seconds=self._offset.value)
        start = window["start"] - offset
        end = window["end"] + offset
        weight = self._data.weight[self._data.weight["timestamp"].between(start, end)]

        self._info.value = (
            f"<b>{self._position + 1}/{len(self._windows)}</b> | {window['description']}"
            f"<br>wskazania w oknie: {_weight_range(weight)}"
        )
        rows = [self._data.pins, *(changes for _, changes in self._layers)]

        with self._figure.batch_update():
            self._figure.data[0].update(x=_times(weight["timestamp"]), y=weight["value"].tolist())
            for index, changes in enumerate(rows):
                steps = _steps(changes, window["pin"], start, end)
                base = _row_base(index, len(rows))
                self._figure.data[index + 1].update(x=_times(steps["timestamp"]), y=(base + steps["value"]).tolist())
            self._figure.layout.shapes = [
                _band(mark_start, mark_end, color, layer, len(rows))
                for mark_start, mark_end, color, layer in window["marks"]
            ]
            self._figure.update_xaxes(range=[_time(start), _time(end)])
            self._figure.update_yaxes(range=_weight_axis(weight), row=1, col=1)


def _empty_figure(layer_names: list[str], title: str, width: int) -> go.FigureWidget:
    figure = go.FigureWidget(
        make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.65, 0.35],
            subplot_titles=("Wskazania [kg]", "Sygnał pinu"),
        )
    )
    figure.add_trace(
        go.Scatter(name="waga", mode="lines+markers", marker={"size": 4}, line={"width": 1}), row=1, col=1
    )

    names = ["surowy", *layer_names]
    for name in names:
        figure.add_trace(go.Scatter(name=name, mode="lines", line={"width": 1.5, "shape": "hv"}), row=2, col=1)

    bases = [_row_base(index, len(names)) for index in range(len(names))]
    figure.update_yaxes(
        tickvals=[base + 0.5 for base in bases],
        ticktext=names,
        range=[-0.3, max(bases) + 1.3],
        fixedrange=True,
        row=2,
        col=1,
    )
    figure.update_xaxes(type="date")
    figure.update_layout(
        title=title,
        height=FIGURE_HEIGHT,
        width=width,
        autosize=False,
        hovermode="x unified",
        margin={"l": 120, "r": 20, "t": 80, "b": 40},
    )
    return figure


def _row_base(index: int, row_count: int) -> float:
    return (row_count - 1 - index) * ROW_SPACING


def _clear(figure: go.FigureWidget) -> None:
    with figure.batch_update():
        for trace in figure.data:
            trace.update(x=[], y=[])
        figure.layout.shapes = []


def _steps(changes: pd.DataFrame, pin: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    pin_changes = changes[changes["pin"] == pin]
    before = pin_changes[pin_changes["timestamp"] < start].tail(1).assign(timestamp=start)
    inside = pin_changes[pin_changes["timestamp"].between(start, end)]
    steps = pd.concat([before, inside], ignore_index=True)

    if steps.empty:
        return steps
    return pd.concat([steps, steps.tail(1).assign(timestamp=end)], ignore_index=True)


def _band(
    start: pd.Timestamp, end: pd.Timestamp, color: str, layer: int | None, row_count: int
) -> dict[str, object]:
    band: dict[str, object] = {
        "type": "rect",
        "xref": "x",
        "x0": _time(start),
        "x1": _time(end),
        "fillcolor": color,
        "line": {"width": 0},
        "layer": "below",
    }
    if layer is None:
        return band | {"yref": "paper", "y0": 0, "y1": 1}

    base = _row_base(layer + 1, row_count)
    return band | {"yref": "y2", "y0": base - BAND_MARGIN, "y1": base + 1 + BAND_MARGIN}


def _layer_marks(events: pd.DataFrame, layer: int) -> list[tuple[pd.Timestamp, pd.Timestamp, str, int]]:
    return [
        (event.start, event.end, LAYER_REJECTED_COLOR if event.rejected else LAYER_EVENT_COLOR, layer)
        for event in events.itertuples()
    ]


def _has_role(events: pd.DataFrame, role: str) -> pd.Series:
    return events["role"] == role if role != ALL_ROLES else pd.Series(True, index=events.index)


def _event_color(rejected: bool) -> str:
    return REJECTED_COLOR if rejected else EVENT_COLOR


def _event_description(event: Any) -> str:
    status = "odrzucone" if event.rejected else "przyjęte"
    return (
        f"pin {event.pin} {event.role}, {status}, {event.start:%Y-%m-%d %H:%M:%S},"
        f" {event.duration.total_seconds():.2f} s, impulsy {int(event.pulse_count)},"
        f" udział aktywnego {event.active_share:.2f}"
    )


def _versions_description(events: pd.DataFrame) -> str:
    return "; ".join(
        f"{'odrzucone' if event.rejected else 'przyjęte'} {event.start:%H:%M:%S}"
        f" {event.duration.total_seconds():.2f} s"
        for event in events.itertuples()
    )


def _times(timestamps: pd.Series) -> list[str]:
    return [_time(timestamp) for timestamp in timestamps]


def _time(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")


def _weight_axis(weight: pd.DataFrame) -> list[float] | None:
    if weight.empty:
        return None
    low = weight["value"].min()
    high = weight["value"].max()
    margin = max((high - low) * WEIGHT_MARGIN_SHARE, MIN_WEIGHT_MARGIN_KG)
    return [float(low - margin), float(high + margin)]


def _weight_range(weight: pd.DataFrame) -> str:
    if weight.empty:
        return "brak"
    first = weight["value"].iloc[0]
    last = weight["value"].iloc[-1]
    return (
        f"min {weight['value'].min():.0f} kg, max {weight['value'].max():.0f} kg,"
        f" pierwsze {first:.0f} kg → ostatnie {last:.0f} kg (zmiana {last - first:+.0f} kg)"
    )
