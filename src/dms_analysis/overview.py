import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureWidgetResampler

from dms_analysis.raw import SessionData

PIN_SPACING = 1.5
PIN_HIGH = 1.0
FIGURE_HEIGHT = 1200
CLEANING_FIGURE_HEIGHT = 1000
FIGURE_WIDTH = 1200


def sessions_table(sessions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sesja": sessions["session_id"],
            "od": sessions["first_at"].dt.strftime("%Y-%m-%d %H:%M"),
            "do": sessions["last_at"].dt.strftime("%Y-%m-%d %H:%M"),
            "czas": sessions["duration"].dt.round("1min").astype(str),
            "wskazania": sessions["weight_count"].fillna(0).astype(int),
            "zmiany pinów": sessions["pin_count"].fillna(0).astype(int),
            "temperatury": sessions["temperature_count"].fillna(0).astype(int),
        }
    ).set_index("sesja")


def changing_pins(pins: pd.DataFrame) -> list[int]:
    distinct_values = pins.groupby("pin")["value"].nunique()
    return sorted(distinct_values[distinct_values > 1].index)


def session_figure(
    data: SessionData,
    title: str,
    pins: list[int] | None = None,
    width: int = FIGURE_WIDTH,
) -> FigureWidgetResampler:
    figure = FigureWidgetResampler(
        make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.55, 0.2, 0.25],
            subplot_titles=("Wskazania [kg]", "Sygnały (piny)", "Temperatury [°C]"),
        )
    )

    _add_weight(figure, data.weight)
    shown_pins = data.pins if pins is None else data.pins[data.pins["pin"].isin(pins)]
    _add_pins(figure, shown_pins, data.end, row=2)
    _add_temperatures(figure, data.temperatures)

    figure.update_layout(
        title=title,
        height=FIGURE_HEIGHT,
        width=width,
        autosize=False,
        hovermode="x unified",
        legend={"groupclick": "toggleitem"},
        margin={"l": 60, "r": 20, "t": 80, "b": 40},
    )
    return figure


def _add_weight(figure: FigureWidgetResampler, weight: pd.DataFrame) -> None:
    figure.add_trace(
        go.Scattergl(name="waga", mode="lines", line={"width": 1}, legendgroup="waga"),
        hf_x=weight["timestamp"],
        hf_y=weight["value"],
        row=1,
        col=1,
    )


def cleaning_figure(
    data: SessionData,
    cleaned: pd.DataFrame,
    title: str,
    pins: list[int],
    width: int = FIGURE_WIDTH,
) -> FigureWidgetResampler:
    figure = FigureWidgetResampler(
        make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=("Wskazania [kg]", "Sygnały surowe", "Sygnały po obróbce"),
        )
    )

    _add_weight(figure, data.weight)
    _add_pins(figure, data.pins[data.pins["pin"].isin(pins)], data.end, row=2, label="surowy pin")
    _add_pins(figure, cleaned[cleaned["pin"].isin(pins)], data.end, row=3, label="pin")

    figure.update_layout(
        title=title,
        height=CLEANING_FIGURE_HEIGHT,
        width=width,
        autosize=False,
        hovermode="x unified",
        legend={"groupclick": "toggleitem"},
        margin={"l": 60, "r": 20, "t": 80, "b": 40},
    )
    return figure


def _add_pins(
    figure: FigureWidgetResampler,
    pins: pd.DataFrame,
    end: pd.Timestamp,
    row: int,
    label: str = "pin",
) -> None:
    pin_numbers = sorted(pins["pin"].unique())

    for position, pin in enumerate(pin_numbers):
        changes = _held_until(pins[pins["pin"] == pin], end)
        base = position * PIN_SPACING
        figure.add_trace(
            go.Scattergl(
                name=f"{label} {pin}",
                mode="lines",
                line={"width": 1, "shape": "hv"},
                legendgroup=label,
            ),
            hf_x=changes["timestamp"],
            hf_y=base + changes["value"] * PIN_HIGH,
            max_n_samples=len(changes),
            row=row,
            col=1,
        )

    figure.update_yaxes(
        tickvals=[position * PIN_SPACING for position in range(len(pin_numbers))],
        ticktext=[f"pin {pin}" for pin in pin_numbers],
        range=[-0.5, max(len(pin_numbers) - 1, 0) * PIN_SPACING + PIN_HIGH + 0.5],
        fixedrange=True,
        row=row,
        col=1,
    )


def _held_until(changes: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    last = changes.iloc[[-1]].assign(timestamp=end)
    return pd.concat([changes, last], ignore_index=True)


def _add_temperatures(figure: FigureWidgetResampler, temperatures: pd.DataFrame) -> None:
    for address, readings in temperatures.groupby("sensor_address"):
        figure.add_trace(
            go.Scattergl(
                name=f"czujnik …{str(address)[-4:]}",
                mode="lines",
                line={"width": 1},
                legendgroup="temperatury",
            ),
            hf_x=readings["timestamp"],
            hf_y=readings["value"],
            row=3,
            col=1,
        )
