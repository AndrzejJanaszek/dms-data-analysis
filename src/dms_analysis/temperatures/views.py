import sqlite3
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dms_analysis import style
from dms_analysis.data.check import TIMEZONE
from dms_analysis.data.paths import prepared_path, temperatures_path

SENSORS = (1, 2, 3, 4)
SENSOR_COLORS = {1: style.BLUE, 2: style.ORANGE, 3: style.AQUA, 4: style.VIOLET}
WINDOW_COLORS = {30: style.RED, 60: style.TEXT}
WINDOWS = (30, 60)


@dataclass(frozen=True)
class Line:
    frame: pd.DataFrame
    column: str
    name: str
    color: str
    width: float = 1.3


@dataclass(frozen=True)
class Panel:
    title: str
    unit: str
    lines: list[Line]


def load_range(full_id: str, start: str, end: str, step_s: int = 1) -> pd.DataFrame:
    return _read(temperatures_path(full_id), "SELECT * FROM temperatures", start, end, step_s)


def load_weight(full_id: str, start: str, end: str, step_s: int = 1) -> pd.DataFrame:
    return _read(prepared_path(full_id), "SELECT timestamp, value FROM weight", start, end, step_s)


def _read(path: object, query: str, start: str, end: str, step_s: int) -> pd.DataFrame:
    begin = pd.Timestamp(start, tz=TIMEZONE).timestamp()
    finish = pd.Timestamp(end, tz=TIMEZONE).timestamp()

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        frame = pd.read_sql_query(
            f"{query} WHERE timestamp >= ? AND timestamp < ? AND CAST(timestamp AS INTEGER) % ? = 0",
            connection,
            params=(begin, finish, step_s),
        )
    finally:
        connection.close()

    moments = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    return frame.assign(local=moments.dt.tz_convert(TIMEZONE).dt.tz_localize(None), utc=moments.dt.strftime("%H:%M:%S UTC"))


def all_figures(
    frame: pd.DataFrame, corners: dict[int, str], title: str, weight: pd.DataFrame | None = None
) -> dict[str, go.Figure]:
    if weight is None:
        top = Panel("temperatura", "°C", temperature_lines(frame, corners))
        prefix = "temperatura"
    else:
        top = Panel("wskazanie", "kg", [Line(weight, "value", "wskazanie", style.WEIGHT)])
        prefix = "wskazanie"

    mean = Line(frame, "mean", "średnia", style.TEXT, width=2.5)
    spread = Line(frame, "spread", "rozrzut (odch. std)", style.RED)
    spread_range = Panel(
        "rozrzut i rozstęp", "°C", [spread, Line(frame, "range", "rozstęp (max − min)", style.TEXT)]
    )
    rates = [
        Panel(f"pochodna: {sensor_name(number, corners)}", "°C/min", window_lines(frame, str(number)))
        for number in SENSORS
    ]

    with_mean = (
        stacked(title, [Panel("temperatura i średnia", "°C", [*top.lines, mean])])
        if weight is None
        else stacked(title, [top, Panel("średnia temperatur", "°C", [mean])])
    )

    return {
        f"{prefix} a średnia": with_mean,
        f"{prefix} a rozrzut": stacked(title, [top, Panel("rozrzut", "°C", [spread])]),
        f"{prefix} a min, max, rozstęp": stacked(
            title,
            [
                top,
                Panel(
                    "max i min", "°C",
                    [Line(frame, "max", "max", style.RED), Line(frame, "min", "min", style.YELLOW)],
                ),
                Panel("rozstęp", "°C", [Line(frame, "range", "rozstęp", style.TEXT)]),
            ],
        ),
        "rozrzut a rozstęp": stacked(title, [spread_range] if weight is None else [top, spread_range]),
        f"{prefix} a pochodne czujników": stacked(title, [top, *rates], match_from=2),
        f"{prefix} a średnia pochodnych": stacked(
            title, [top, Panel("średnia pochodnych", "°C/min", window_lines(frame, "mean"))]
        ),
        f"{prefix} a odchylenie std pochodnych": stacked(
            title, [top, Panel("odchylenie std pochodnych", "°C/min", window_lines(frame, "spread"))]
        ),
    }


def temperature_lines(frame: pd.DataFrame, corners: dict[int, str]) -> list[Line]:
    return [Line(frame, f"t{number}", sensor_name(number, corners), SENSOR_COLORS[number]) for number in SENSORS]


def window_lines(frame: pd.DataFrame, name: str) -> list[Line]:
    return [Line(frame, f"rate{window}_{name}", f"okno {window} s", WINDOW_COLORS[window]) for window in WINDOWS]


def stacked(title: str, panels: list[Panel], match_from: int | None = None) -> go.Figure:
    rows = len(panels)
    titles = [f"{title} — {panels[0].title}", *(panel.title for panel in panels[1:])]
    figure = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=titles)
    figure.update_layout(height=260 + 230 * rows, hovermode="x unified")
    figure.update_xaxes(title_text="czas lokalny", row=rows, col=1)

    shown: set[str] = set()
    for row, panel in enumerate(panels, start=1):
        for line in panel.lines:
            add_line(figure, line, row, panel.unit, legend=line.name not in shown)
            shown.add(line.name)
        figure.update_yaxes(title_text=panel.unit, row=row, col=1)
        if match_from is not None and row > match_from:
            figure.update_yaxes(matches=f"y{match_from}", row=row, col=1)
    return figure


def add_line(figure: go.Figure, line: Line, row: int, unit: str, legend: bool) -> None:
    figure.add_trace(
        go.Scatter(
            x=line.frame["local"], y=line.frame[line.column], name=line.name, legendgroup=line.name,
            showlegend=legend, line={"color": line.color, "width": line.width}, customdata=line.frame["utc"],
            hovertemplate=f"%{{y:.3f}} {unit} (%{{customdata}})",
        ),
        row=row, col=1,
    )


def sensor_name(number: int, corners: dict[int, str]) -> str:
    corner = corners.get(number)
    return f"czujnik {number} ({corner})" if corner else f"czujnik {number}"
