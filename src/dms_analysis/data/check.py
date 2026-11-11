import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dms_analysis.data.catalog import Catalog
from dms_analysis.data.paths import prepared_path
from dms_analysis.data.prepare import epoch_seconds, is_stale
from dms_analysis.export import EVENT, RAW_KIND, REJECTED
from dms_analysis.raw import TIMEZONE

ENVELOPE_BINS = 3000
DAY_TEMPERATURE_BINS = 10000
CHANGE_WINDOW_S = 3
MANUAL_MARGIN = pd.Timedelta(minutes=15)
PIN_SPACING = 1.5
WIDTH = 1250

WEIGHT_COLOR = "#264653"
EVENT_COLOR = "rgba(42, 157, 143, 0.18)"
MANUAL_COLOR = "rgba(233, 196, 106, 0.35)"
ISSUE_COLOR = "rgba(231, 111, 81, 0.18)"
FULL_DAY = "pełna"
FULL_DAY_WITH_ISSUE = "pełna z problemem"
PARTIAL_DAY = "niepełna"
DAY_COLORS = {FULL_DAY: "#2a9d8f", FULL_DAY_WITH_ISSUE: "#f4a261", PARTIAL_DAY: "#c9c9c9"}
DAY_MARKS = {FULL_DAY: " ✓", FULL_DAY_WITH_ISSUE: " ⚠", PARTIAL_DAY: ""}

TIME_STATUS = {"real": "czas rzeczywisty", "estimated": "czas szacowany", "unknown": "czas zegara urządzenia"}
STATE_LEVELS = {0: 0.0, EVENT: 1.0, REJECTED: 0.5}


@dataclass(frozen=True)
class PreparedPeriod:
    info: pd.Series
    weight: pd.DataFrame
    temperatures: pd.DataFrame
    removed: pd.DataFrame
    changes: pd.DataFrame
    events: pd.DataFrame
    issues: pd.DataFrame
    parts: pd.DataFrame

    @property
    def start(self) -> pd.Timestamp:
        return self.weight["timestamp"].min()

    @property
    def end(self) -> pd.Timestamp:
        return self.weight["timestamp"].max()

    @property
    def title(self) -> str:
        return str(self.info["label"])


def load_period(full_id: str) -> PreparedPeriod:
    path = prepared_path(full_id)
    if not path.exists():
        raise FileNotFoundError(f"brak {path} — uruchom prepare")

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = {
            name: pd.read_sql_query(f"SELECT * FROM {name}", connection)
            for name in (
                "period",
                "weight",
                "temperatures",
                "removed",
                "signal_changes",
                "signal_events",
                "issues",
                "parts",
            )
        }
    finally:
        connection.close()

    return PreparedPeriod(
        info=tables["period"].iloc[0],
        weight=_local(tables["weight"], "timestamp"),
        temperatures=_local(tables["temperatures"], "timestamp"),
        removed=_local(tables["removed"], "timestamp"),
        changes=_local(tables["signal_changes"], "timestamp"),
        events=_local(tables["signal_events"], "start", "end"),
        issues=_local(tables["issues"], "start", "end"),
        parts=_local(tables["parts"], "start", "end"),
    )


def _local(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    return frame.assign(
        **{column: pd.to_datetime(frame[column], unit="s", utc=True).dt.tz_convert(TIMEZONE) for column in columns}
    )


def staleness_warning(catalog: Catalog, full_id: str) -> str | None:
    if is_stale(catalog.period(full_id)):
        return "katalog zmieniony od ostatniego prepare — uruchom prepare (uv run python -m dms_analysis.data prepare)"
    return None


def description_table(period: PreparedPeriod) -> pd.DataFrame:
    info = period.info
    source = f"{info['dump']}, " + (
        f"sesja {info['session']}" if pd.notna(info["session"]) else f"id {info['first_id']}–{info['last_id']}"
    )
    rows = {
        "okres": info["id"],
        "dawna nazwa": info["legacy_name"],
        "źródło": source,
        "sprzęt": info["hardware"],
        "czas": time_description(info),
        "początek (katalog)": _moment(info["start"]),
        "koniec (katalog)": _moment(info["end"]),
        "pierwsze wskazanie": f"{period.start:%Y-%m-%d %H:%M:%S %z}",
        "ostatnie wskazanie": f"{period.end:%Y-%m-%d %H:%M:%S %z}",
        "wskazania": f"{len(period.weight):,}".replace(",", " "),
        "kanał waga": info["channel_weight"],
        "kanał odsyp": info["channel_dispense"],
        "kanał zasyp": info["channel_refill"],
        "kanał temperatury": info["channel_temperatures"],
        "sprawdzony": "tak" if info["checked"] else "nie",
        "wariant reguł sygnałów": f"{info['variant']} ({info['variant_description']})",
        "przygotowano": f"{info['prepared_at']}, commit {info['commit']}",
        "uwagi": info["note"],
    }
    return pd.DataFrame({"wartość": rows})


def time_description(info: pd.Series) -> str:
    status = TIME_STATUS[info["time"]]
    if info["time"] == "estimated":
        return f"{status}, przesunięcie {info['clock_offset_h']:+g} h, niepewność ±{info['time_uncertainty_h']:g} h"
    if info["time"] == "unknown":
        return f"{status} (rzeczywisty nieznany) — godziny i daty nie nadają się do analiz zależnych od pory dnia"
    return status


def _moment(seconds: float) -> str:
    return f"{pd.Timestamp(seconds, unit='s', tz='UTC').tz_convert(TIMEZONE):%Y-%m-%d %H:%M:%S %z}"


def overview_figure(period: PreparedPeriod) -> go.Figure:
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.15, 0.15, 0.3],
        subplot_titles=(
            "Wskazania [kg] (min i max w przedziale)",
            "Sygnały surowe",
            "Sygnały oczyszczone (zdarzenie = 1, odrzucone krótkie = 0,5; zasypy wpisane ręcznie zaznaczone na żółto)",
            "Temperatury [°C] (bez usuniętych odczytów)",
        ),
    )
    _weight_trace(figure, period.weight, 1, ENVELOPE_BINS)
    _signal_traces(figure, period.changes[period.changes["kind"] == RAW_KIND], period.end, 2, raw=True)
    _signal_traces(figure, period.changes[period.changes["kind"] != RAW_KIND], period.end, 3, raw=False)
    _temperature_traces(figure, period.temperatures, 4, ENVELOPE_BINS)
    _manual_bands(figure, period)
    _issue_bands(figure, period)
    _fill_empty_panels(figure, period, period.start, period.end)
    _layout(figure, f"<b>{period.title}</b> · cały okres", 1000)
    return figure


def _weight_trace(figure: go.Figure, weight: pd.DataFrame, row: int, bins: int) -> None:
    if weight.empty:
        return
    reduced = envelope(weight, "value", bins)
    figure.add_trace(
        go.Scatter(
            x=_plot_time(reduced["timestamp"]),
            y=reduced["value"],
            name="wskazanie",
            mode="lines",
            line={"width": 1, "color": WEIGHT_COLOR},
        ),
        row=row,
        col=1,
    )


def _signal_traces(figure: go.Figure, changes: pd.DataFrame, end: pd.Timestamp, row: int, raw: bool) -> None:
    pins = sorted(changes["pin"].unique())
    for position, pin in enumerate(pins):
        pin_changes = changes[changes["pin"] == pin].sort_values("timestamp")
        levels = pin_changes["state"] if raw else pin_changes["state"].map(STATE_LEVELS)
        held = pd.concat(
            [pin_changes.assign(level=levels), pin_changes.iloc[[-1]].assign(level=levels.iloc[-1], timestamp=end)]
        )
        figure.add_trace(
            go.Scatter(
                x=_plot_time(held["timestamp"]),
                y=position * PIN_SPACING + held["level"],
                name=f"pin {pin} {'surowy' if raw else 'oczyszczony'}",
                mode="lines",
                line={"width": 1, "shape": "hv"},
            ),
            row=row,
            col=1,
        )
    figure.update_yaxes(
        tickvals=[position * PIN_SPACING for position in range(len(pins))],
        ticktext=[f"pin {pin}" for pin in pins],
        row=row,
        col=1,
    )


def _temperature_traces(figure: go.Figure, temperatures: pd.DataFrame, row: int, bins: int) -> None:
    for sensor, readings in temperatures.groupby("sensor"):
        reduced = envelope(readings.reset_index(drop=True), "value", bins)
        figure.add_trace(
            go.Scatter(
                x=_plot_time(reduced["timestamp"]),
                y=reduced["value"],
                name=f"czujnik {sensor}",
                mode="lines",
                line={"width": 1},
            ),
            row=row,
            col=1,
        )


def _manual_bands(figure: go.Figure, period: PreparedPeriod) -> None:
    manual = period.events[period.events["synthetic"] == 1]
    for event in manual.itertuples():
        figure.add_vrect(
            x0=_plot_moment(event.start),
            x1=_plot_moment(event.end),
            fillcolor=MANUAL_COLOR,
            line_width=0,
            row="all",
            col=1,
        )


def _issue_bands(figure: go.Figure, period: PreparedPeriod) -> None:
    for issue in period.issues.itertuples():
        figure.add_vrect(
            x0=_plot_moment(issue.start),
            x1=_plot_moment(issue.end),
            fillcolor=ISSUE_COLOR,
            line_width=0,
            annotation_text=f"problem: {issue.channel}, {issue.kind}",
            annotation_position="top left",
            row="all",
            col=1,
        )


def _fill_empty_panels(figure: go.Figure, period: PreparedPeriod, start: pd.Timestamp, end: pd.Timestamp) -> None:
    rows_with_data = {trace.yaxis for trace in figure.data}
    for row in range(1, 5):
        axis = "y" if row == 1 else f"y{row}"
        if axis in rows_with_data:
            continue
        figure.add_trace(
            go.Scatter(
                x=[_plot_moment(start), _plot_moment(end)],
                y=[0, 0],
                mode="markers",
                marker={"opacity": 0},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        figure.update_yaxes(showticklabels=False, row=row, col=1)
        reference = "" if row == 1 else str(row)
        figure.add_annotation(
            text="brak danych",
            xref=f"x{reference} domain",
            yref=f"y{reference} domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16, "color": "#999"},
        )


def _layout(figure: go.Figure, title: str, height: int) -> None:
    figure.update_xaxes(type="date", showticklabels=True, tickformat="%d.%m\n%H:%M")
    figure.update_layout(
        title=title,
        height=height,
        width=WIDTH,
        hovermode="x unified",
        margin={"l": 70, "r": 20, "t": 90, "b": 40},
        legend={"orientation": "h", "y": -0.05},
    )


def envelope(frame: pd.DataFrame, column: str, bins: int) -> pd.DataFrame:
    if len(frame) <= 2 * bins:
        return frame
    seconds = frame["timestamp"].astype("int64")
    bucket = pd.cut(seconds, bins, labels=False)
    grouped = frame[column].groupby(bucket)
    chosen = pd.Index(grouped.idxmin()).union(pd.Index(grouped.idxmax()))
    return frame.loc[chosen].sort_values("timestamp")


def _plot_time(moments: pd.Series) -> pd.Series:
    return moments.dt.tz_localize(None)


def _plot_moment(moment: pd.Timestamp) -> pd.Timestamp:
    return moment.tz_localize(None)


def weight_changes(period: PreparedPeriod) -> pd.DataFrame:
    events = period.events[period.events["rejected"] == 0]
    seconds = epoch_seconds(period.weight["timestamp"]).to_numpy()
    values = period.weight["value"].to_numpy()

    def median_between(start: float, end: float) -> float:
        first, last = np.searchsorted(seconds, [start, end])
        return float(np.median(values[first:last])) if last > first else np.nan

    starts = epoch_seconds(events["start"]).to_numpy()
    ends = epoch_seconds(events["end"]).to_numpy()
    change = [
        median_between(end, end + CHANGE_WINDOW_S) - median_between(start - CHANGE_WINDOW_S, start)
        for start, end in zip(starts, ends, strict=True)
    ]
    return events.assign(change=change)


def suspicious_events(period: PreparedPeriod) -> pd.DataFrame:
    events = weight_changes(period)
    dispense_without_drop = (events["role"] == "odsyp") & (events["change"] >= 0)
    refill_without_rise = (events["role"] == "zasyp") & (events["change"] <= 0)
    suspicious = events[dispense_without_drop | refill_without_rise | events["change"].isna()]
    return pd.DataFrame(
        {
            "rola": suspicious["role"],
            "początek": suspicious["start"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "czas [s]": suspicious["duration"].round(1),
            "impulsy": suspicious["pulse_count"],
            "ręczne": suspicious["synthetic"].map({0: "", 1: "tak"}),
            "zmiana mediany 3 s [kg]": suspicious["change"],
        }
    )


def event_figure(period: PreparedPeriod, start: pd.Timestamp, end: pd.Timestamp, title: str) -> go.Figure:
    weight = period.weight[(period.weight["timestamp"] >= start) & (period.weight["timestamp"] <= end)]
    changes = _window_changes(period.changes, start, end)

    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.6, 0.2, 0.2])
    figure.add_trace(
        go.Scatter(
            x=_plot_time(weight["timestamp"]),
            y=weight["value"],
            name="wskazanie",
            mode="lines+markers",
            marker={"size": 3},
            line={"width": 1, "color": WEIGHT_COLOR},
        ),
        row=1,
        col=1,
    )
    _signal_traces(figure, changes[changes["kind"] == RAW_KIND], end, 2, raw=True)
    _signal_traces(figure, changes[changes["kind"] != RAW_KIND], end, 3, raw=False)

    inside = period.events[(period.events["end"] >= start) & (period.events["start"] <= end)]
    for event in inside.itertuples():
        color = MANUAL_COLOR if event.synthetic else EVENT_COLOR
        figure.add_vrect(
            x0=_plot_moment(max(event.start, start)),
            x1=_plot_moment(min(event.end, end)),
            fillcolor=color,
            line_width=0,
            row="all",
            col=1,
        )
    figure.update_xaxes(range=[_plot_moment(start), _plot_moment(end)])
    _layout(figure, title, 650)
    return figure


def _window_changes(changes: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    before = changes[changes["timestamp"] < start].sort_values("timestamp").groupby(["pin", "kind"]).tail(1)
    inside = changes[(changes["timestamp"] >= start) & (changes["timestamp"] <= end)]
    return pd.concat([before.assign(timestamp=start), inside], ignore_index=True).sort_values("timestamp")


def suspicious_figure(period: PreparedPeriod, number: int, margin_s: float = 120) -> go.Figure:
    events = weight_changes(period)
    suspicious = events[
        ((events["role"] == "odsyp") & (events["change"] >= 0))
        | ((events["role"] == "zasyp") & (events["change"] <= 0))
        | events["change"].isna()
    ]
    event = suspicious.iloc[number]
    margin = pd.Timedelta(seconds=margin_s)
    title = (
        f"<b>{period.title}</b> · {event['role']} {event['start']:%Y-%m-%d %H:%M:%S}, zmiana {event['change']:+.0f} kg"
    )
    return event_figure(period, event["start"] - margin, event["end"] + margin, title)


def manual_event_figures(period: PreparedPeriod) -> list[go.Figure]:
    manual = period.events[period.events["synthetic"] == 1]
    return [
        event_figure(
            period,
            event.start - MANUAL_MARGIN,
            event.end + MANUAL_MARGIN,
            f"<b>{period.title}</b> · ręczny {event.role} {event.start:%Y-%m-%d %H:%M:%S} – {event.end:%H:%M:%S}"
            + (" (otwarty do końca okresu)" if event.end == period.end else ""),
        )
        for event in manual.itertuples()
    ]


def removed_summary(period: PreparedPeriod) -> pd.DataFrame:
    if period.removed.empty:
        return pd.DataFrame({"liczba": []})
    return (
        period.removed.groupby(["kind", "sensor_address"])
        .agg(
            liczba=("value", "size"),
            wartości=("value", lambda values: ", ".join(f"{value:g}" for value in sorted(set(values)))),
        )
        .rename_axis(["rodzaj", "adres czujnika"])
    )


def removed_figure(period: PreparedPeriod) -> go.Figure:
    figure = go.Figure()
    _weight_trace_single(figure, period)
    for kind, rows in period.removed.groupby("kind"):
        figure.add_trace(
            go.Scatter(
                x=_plot_time(rows["timestamp"]),
                y=[period.weight["value"].max()] * len(rows),
                name=f"usunięte: {kind}",
                mode="markers",
                marker={"size": 9, "symbol": "x"},
                text=rows["sensor_address"] + " = " + rows["value"].map("{:g}".format),
            )
        )
    _layout(figure, f"<b>{period.title}</b> · gdzie w czasie usunięto odczyty temperatur", 400)
    return figure


def _weight_trace_single(figure: go.Figure, period: PreparedPeriod) -> None:
    reduced = envelope(period.weight, "value", ENVELOPE_BINS)
    figure.add_trace(
        go.Scatter(
            x=_plot_time(reduced["timestamp"]),
            y=reduced["value"],
            name="wskazanie",
            mode="lines",
            line={"width": 1, "color": WEIGHT_COLOR},
        )
    )


def parts_table(period: PreparedPeriod) -> pd.DataFrame:
    steps = period.weight["timestamp"].diff().dt.total_seconds()
    return pd.DataFrame(
        {
            "część": period.parts["part"],
            "początek": period.parts["start"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "koniec": period.parts["end"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "godziny": ((period.parts["end"] - period.parts["start"]).dt.total_seconds() / 3600).round(1),
            "najdłuższa przerwa w okresie [s]": round(float(steps.max()), 1),
        }
    )


def full_days(parts: list[tuple[pd.Timestamp, pd.Timestamp]], timezone: str = TIMEZONE) -> list[date]:
    days = []
    for start, end in parts:
        day = start.tz_convert(timezone).date()
        while day <= end.tz_convert(timezone).date():
            midnight = pd.Timestamp(day).tz_localize(timezone)
            next_midnight = pd.Timestamp(day + timedelta(days=1)).tz_localize(timezone)
            if start <= midnight and next_midnight <= end:
                days.append(day)
            day += timedelta(days=1)
    return days


def period_days(period: PreparedPeriod) -> list[date]:
    if period.info["time"] == "unknown":
        return []
    parts = list(zip(period.parts["start"], period.parts["end"], strict=True))
    return full_days(parts)


def day_bounds(day: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(day).tz_localize(TIMEZONE), pd.Timestamp(day + timedelta(days=1)).tz_localize(TIMEZONE)


def calendar_days(period: PreparedPeriod) -> list[date]:
    days = []
    day = period.start.date()
    while day <= period.end.date():
        days.append(day)
        day += timedelta(days=1)
    return days


def day_issues(period: PreparedPeriod, day: date) -> list[str]:
    midnight, next_midnight = day_bounds(day)
    overlapping = period.issues[(period.issues["start"] < next_midnight) & (period.issues["end"] > midnight)]
    return [f"{issue.channel}: {issue.kind}" for issue in overlapping.itertuples()]


def day_status(period: PreparedPeriod, day: date, complete: set[date]) -> str:
    if day not in complete:
        return PARTIAL_DAY
    return FULL_DAY_WITH_ISSUE if day_issues(period, day) else FULL_DAY


def days_table(period: PreparedPeriod) -> pd.DataFrame:
    if period.info["time"] == "unknown":
        return pd.DataFrame({"doba": []})
    complete = set(period_days(period))
    rows = []
    for day in calendar_days(period):
        midnight, next_midnight = day_bounds(day)
        covered = min(next_midnight, period.end) - max(midnight, period.start)
        rows.append(
            {
                "doba": f"{day:%Y-%m-%d %a}",
                "status": day_status(period, day, complete),
                "godziny w okresie": round(covered.total_seconds() / 3600, 1),
                "długość doby [h]": round((next_midnight - midnight).total_seconds() / 3600),
                "problemy z katalogu": ", ".join(day_issues(period, day)),
            }
        )
    return pd.DataFrame(rows)


def days_figure(period: PreparedPeriod) -> go.Figure:
    complete = set(period_days(period))
    figure = go.Figure()
    _weight_trace_single(figure, period)
    if period.info["time"] == "unknown":
        _layout(figure, f"<b>{period.title}</b> · czas zegara urządzenia — dób nie wyznaczamy", 420)
        return figure

    for day in calendar_days(period):
        midnight, next_midnight = day_bounds(day)
        status = day_status(period, day, complete)
        figure.add_vrect(
            x0=_plot_moment(max(midnight, period.start)),
            x1=_plot_moment(min(next_midnight, period.end)),
            fillcolor=DAY_COLORS[status],
            opacity=0.2,
            line_width=1,
            line_color="white",
            annotation_text=f"{day:%d.%m}{DAY_MARKS[status]}",
            annotation_position="top left",
        )

    for issue in period.issues.itertuples():
        figure.add_vrect(
            x0=_plot_moment(issue.start), x1=_plot_moment(issue.end), fillcolor=ISSUE_COLOR, opacity=1, line_width=0
        )
    for part in period.parts.itertuples():
        figure.add_vline(x=_plot_moment(part.start), line_dash="dot", line_color="#555")
    _layout(
        figure,
        f"<b>{period.title}</b> · doby pełne (zielone, ✓), pełne z problemem z katalogu (pomarańczowe, ⚠)"
        " i niepełne (szare); pasma = problemy; linie kropkowane = początki części",
        420,
    )
    return figure


def day_figure(period: PreparedPeriod, day: date) -> go.Figure:
    start = max(pd.Timestamp(day).tz_localize(TIMEZONE), period.start)
    end = min(pd.Timestamp(day + timedelta(days=1)).tz_localize(TIMEZONE), period.end)
    weight = period.weight[(period.weight["timestamp"] >= start) & (period.weight["timestamp"] < end)]
    temperatures = period.temperatures[
        (period.temperatures["timestamp"] >= start) & (period.temperatures["timestamp"] < end)
    ]
    changes = _window_changes(period.changes, start, end)

    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.15, 0.15, 0.3],
        subplot_titles=(
            "Wskazania [kg] (każdy odczyt)",
            "Sygnały surowe",
            "Sygnały oczyszczone",
            "Temperatury [°C] (min i max w przedziałach ok. 9 s)",
        ),
    )
    figure.add_trace(
        go.Scatter(
            x=_plot_time(weight["timestamp"]),
            y=weight["value"],
            name="wskazanie",
            mode="lines",
            line={"width": 1, "color": WEIGHT_COLOR},
        ),
        row=1,
        col=1,
    )
    _signal_traces(figure, changes[changes["kind"] == RAW_KIND], end, 2, raw=True)
    _signal_traces(figure, changes[changes["kind"] != RAW_KIND], end, 3, raw=False)
    _temperature_traces(figure, temperatures, 4, DAY_TEMPERATURE_BINS)
    _fill_empty_panels(figure, period, start, end)
    figure.update_xaxes(range=[_plot_moment(start), _plot_moment(end)])
    _layout(figure, f"<b>{period.title}</b> · doba {day:%Y-%m-%d} w pełnej rozdzielczości", 1000)
    return figure
