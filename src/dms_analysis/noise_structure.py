from collections.abc import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dms_analysis.between_events_fragments import NO_FRAGMENT, RESOLUTION_KG
from dms_analysis.synthetic import ALL, DAY, NIGHT, part_of_day
from dms_analysis.weight_changes import residual_maker, span_fit

WHITE = "biały szum"
PARTS = (NIGHT, DAY, ALL)
MAX_JUMP_STEPS = 5
DETREND_SPAN_S = 60
MAX_LAG_S = 30
SEGMENT_S = 1200
PERIOD_BANDS_S = ((2, 10), (10, 30), (30, 60), (60, 300), (300, SEGMENT_S))
FIGURE_HEIGHT = 450


def jumps(samples: pd.DataFrame) -> pd.DataFrame:
    inside = samples[samples["fragment"] != NO_FRAGMENT]
    values = inside["value"].where(~inside["spike"])
    steps = values.groupby(inside["fragment"]).diff() / RESOLUTION_KG

    table = pd.DataFrame({"timestamp": inside["timestamp"], "steps": steps}).dropna()
    table["steps"] = table["steps"].round().astype(int)
    table["part_of_day"] = table["timestamp"].dt.hour.map(part_of_day)
    return table


def white_sigma_kg(jump_table: pd.DataFrame) -> float:
    variance = (jump_table["steps"] * RESOLUTION_KG).var() / 2
    return float(np.sqrt(variance - RESOLUTION_KG**2 / 12))


def jump_summary(jump_table: pd.DataFrame, white_table: pd.DataFrame) -> pd.DataFrame:
    columns = {part: _jump_shares(jump_table, part) for part in PARTS}
    columns[WHITE] = _jump_shares(white_table, ALL)
    labels = [_jump_label(steps) for steps in range(MAX_JUMP_STEPS + 2)]
    return pd.DataFrame(columns).reindex(labels).fillna(0.0)


def jump_figure(summary: pd.DataFrame, width: int) -> go.Figure:
    figure = go.Figure()
    for column in [NIGHT, DAY, WHITE]:
        figure.add_trace(go.Bar(name=column, x=summary.index, y=summary[column]))

    figure.update_xaxes(title_text="skok wskazania między kolejnymi odczytami (co ok. 1 s)")
    figure.update_yaxes(type="log", title_text="udział odczytów [%]")
    figure.update_layout(
        title="Jak często wskazanie skacze o daną liczbę działek (20 kg)",
        barmode="group",
        width=width,
        height=FIGURE_HEIGHT,
    )
    return figure


def residual_windows(samples: pd.DataFrame, span_s: int) -> pd.DataFrame:
    inside = samples[samples["fragment"] != NO_FRAGMENT]
    values = inside["value"].where(~inside["spike"])
    tables = []

    for _, fragment in values.groupby(inside["fragment"]):
        residuals = (fragment - span_fit(fragment, span_s, 1)).to_numpy()
        count = len(residuals) // span_s
        windows = residuals[: count * span_s].reshape(count, span_s)
        complete = ~np.isnan(windows).any(axis=1)
        starts = fragment.index[: count * span_s : span_s][complete]
        tables.append(
            pd.DataFrame(
                {
                    "part_of_day": inside.loc[starts, "timestamp"].dt.hour.map(part_of_day).to_numpy(),
                    "residuals": list(windows[complete]),
                }
            )
        )

    return pd.concat(tables, ignore_index=True)


def autocorrelation(windows: pd.DataFrame, white_windows: pd.DataFrame) -> pd.DataFrame:
    return _per_part(windows, white_windows, _autocorrelation)


def autocorrelation_figure(table: pd.DataFrame, width: int) -> go.Figure:
    figure = go.Figure()
    for column in table.columns:
        dash = "dash" if column == WHITE else "solid"
        figure.add_trace(go.Scatter(name=column, x=table.index, y=table[column], mode="lines", line={"dash": dash}))

    figure.add_hline(y=0, line={"color": "gray", "width": 1})
    figure.update_xaxes(title_text="przesunięcie [s]")
    figure.update_yaxes(title_text="autokorelacja reszt")
    figure.update_layout(
        title=f"Autokorelacja reszt po odjęciu prostej w odcinkach {DETREND_SPAN_S} s",
        width=width,
        height=FIGURE_HEIGHT,
    )
    return figure


def segments(samples: pd.DataFrame) -> pd.DataFrame:
    inside = samples[samples["fragment"] != NO_FRAGMENT]
    values = inside["value"].where(~inside["spike"])
    step = SEGMENT_S // 2
    maker = residual_maker(SEGMENT_S, 1)
    rows = []

    for _, run in values.groupby([inside["fragment"], values.isna().cumsum()]):
        run = run.dropna()
        for start in range(0, len(run) - SEGMENT_S + 1, step):
            rows.append(
                {
                    "part_of_day": part_of_day(inside.loc[run.index[start], "timestamp"].hour),
                    "residuals": run.to_numpy()[start : start + SEGMENT_S] @ maker.T,
                }
            )

    return pd.DataFrame(rows)


def spectrum(segment_table: pd.DataFrame, white_segments: pd.DataFrame) -> pd.DataFrame:
    table = _per_part(segment_table, white_segments, _power_density)
    table.index = pd.Index(np.fft.rfftfreq(SEGMENT_S), name="frequency_hz")
    return table.iloc[1:]


def band_summary(spectrum_table: pd.DataFrame) -> pd.DataFrame:
    periods = 1 / spectrum_table.index.to_numpy()
    step_hz = spectrum_table.index[1] - spectrum_table.index[0]
    rows = {}
    for shortest, longest in PERIOD_BANDS_S:
        chosen = (periods >= shortest) & (periods < longest)
        rows[f"{shortest}–{longest} s"] = np.sqrt(spectrum_table[chosen].sum() * step_hz)
    return pd.DataFrame(rows).T


def spectrum_figure(spectrum_table: pd.DataFrame, width: int) -> go.Figure:
    figure = go.Figure()
    periods = 1 / spectrum_table.index.to_numpy()
    for column in spectrum_table.columns:
        dash = "dash" if column == WHITE else "solid"
        figure.add_trace(
            go.Scatter(name=column, x=periods, y=spectrum_table[column], mode="lines", line={"dash": dash})
        )

    figure.update_xaxes(type="log", title_text="okres wahania [s]")
    figure.update_yaxes(type="log", title_text="gęstość widmowa mocy [kg²·s]")
    figure.update_layout(
        title=f"Widmo wskazań (Welch, odcinki {SEGMENT_S} s z odjętą prostą)",
        width=width,
        height=FIGURE_HEIGHT,
    )
    return figure


def detrend_figure(samples: pd.DataFrame, fragment: int, start_min: float, minutes: float, width: int) -> go.Figure:
    shown = samples[samples["fragment"] == fragment]
    values = shown["value"].where(~shown["spike"])
    line = span_fit(values, DETREND_SPAN_S, 1)
    chosen = slice(int(start_min * 60), int((start_min + minutes) * 60))
    times = shown["timestamp"].iloc[chosen]

    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    figure.add_trace(
        go.Scatter(name="wskazanie", x=times, y=values.iloc[chosen], mode="lines+markers", marker={"size": 3}),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(name=f"prosta w odcinkach {DETREND_SPAN_S} s", x=times, y=line.iloc[chosen], mode="lines"),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(name="reszta", x=times, y=(values - line).iloc[chosen], mode="lines+markers", marker={"size": 3}),
        row=2,
        col=1,
    )

    figure.update_yaxes(title_text="wskazanie [kg]", row=1, col=1)
    figure.update_yaxes(title_text="reszta [kg]", row=2, col=1)
    figure.update_layout(
        title=f"Fragment {fragment}: wskazania, prosta odejmowana przed autokorelacją i reszty",
        width=width,
        height=2 * FIGURE_HEIGHT,
        hovermode="x unified",
    )
    return figure


def _jump_shares(jump_table: pd.DataFrame, part: str) -> pd.Series:
    chosen = jump_table if part == ALL else jump_table[jump_table["part_of_day"] == part]
    size = chosen["steps"].abs().clip(upper=MAX_JUMP_STEPS + 1)
    shares = 100 * size.value_counts(normalize=True)
    shares.index = [_jump_label(steps) for steps in shares.index]
    return shares


def _jump_label(steps: int) -> str:
    if steps == 0:
        return "0"
    if steps > MAX_JUMP_STEPS:
        return f"> ±{MAX_JUMP_STEPS}"
    return f"±{steps}"


def _per_part(
    windows: pd.DataFrame, white_windows: pd.DataFrame, measure: Callable[[np.ndarray], np.ndarray]
) -> pd.DataFrame:
    columns = {}
    for part in PARTS:
        chosen = windows if part == ALL else windows[windows["part_of_day"] == part]
        columns[part] = measure(np.stack(chosen["residuals"].to_list()))
    columns[WHITE] = measure(np.stack(white_windows["residuals"].to_list()))
    return pd.DataFrame(columns)


def _autocorrelation(residuals: np.ndarray) -> np.ndarray:
    energy = (residuals**2).sum(axis=1)
    moving = residuals[energy > 0]
    energy = energy[energy > 0]
    width = moving.shape[1]
    return np.array(
        [((moving[:, : width - lag] * moving[:, lag:]).sum(axis=1) / energy).mean() for lag in range(MAX_LAG_S + 1)]
    )


def _power_density(residuals: np.ndarray) -> np.ndarray:
    window = np.hanning(residuals.shape[1])
    transformed = np.fft.rfft(residuals * window, axis=1)
    density = 2 * np.abs(transformed) ** 2 / (window**2).sum()
    return density.mean(axis=0)
