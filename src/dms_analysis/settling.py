import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dms_analysis.between_events_fragments import NO_FRAGMENT
from dms_analysis.prepared import PreparedSession

DISPENSE = "odsyp"
BEFORE_S = 30
AFTER_S = 120
REFERENCE_S = 30
STEADY_WINDOW_S = 5
CONTROL_PER_DISPENSE = 5
CONTROL_SEED = 0
STEADY_QUANTILE = 0.9
FIGURE_HEIGHT = 450
BAND_COLOR = "rgba(0, 100, 200, 0.2)"
CONTROL_BAND_COLOR = "rgba(120, 120, 120, 0.2)"
LINE_COLOR = "rgba(0, 100, 200, 0.25)"


def free_dispenses(session: PreparedSession) -> pd.DataFrame:
    events = session.events.sort_values("start").reset_index(drop=True)
    events["free_after_s"] = (events["start"].shift(-1) - events["end"]).dt.total_seconds()

    chosen = (events["role"] == DISPENSE) & ~events["rejected"] & (events["free_after_s"] >= AFTER_S)
    return events[chosen].reset_index(drop=True)


def control_moments(samples: pd.DataFrame, dispenses: pd.DataFrame) -> pd.Series:
    inside = samples[samples["fragment"] != NO_FRAGMENT]
    first = inside.groupby("fragment")["timestamp"].transform("first")
    last = inside.groupby("fragment")["timestamp"].transform("last")
    fits = (inside["timestamp"] - first >= pd.Timedelta(seconds=BEFORE_S)) & (
        last - inside["timestamp"] >= pd.Timedelta(seconds=AFTER_S)
    )
    candidates = inside.loc[fits, "timestamp"]

    generator = np.random.default_rng(CONTROL_SEED)
    moments = []
    for end in dispenses["end"]:
        same_hour = candidates[candidates.dt.hour == end.hour]
        if not same_hour.empty:
            moments.extend(same_hour.iloc[generator.integers(len(same_hour), size=CONTROL_PER_DISPENSE)])
    return pd.Series(moments, dtype=candidates.dtype)


def aligned(session: PreparedSession, moments: pd.Series) -> pd.DataFrame:
    weight = session.data.weight.set_index("timestamp")["value"].astype(float)
    parts = []

    for number, moment in enumerate(moments):
        shown = weight[
            (weight.index >= moment - pd.Timedelta(seconds=BEFORE_S))
            & (weight.index < moment + pd.Timedelta(seconds=AFTER_S))
        ]
        seconds = np.round((shown.index - moment).total_seconds()).astype(int)
        reference = shown[seconds >= AFTER_S - REFERENCE_S].median()
        parts.append(
            pd.DataFrame(
                {
                    "number": number,
                    "moment": moment,
                    "t_s": seconds,
                    "value_kg": shown.to_numpy(),
                    "deviation_kg": shown.to_numpy() - reference,
                    "step_kg": np.abs(np.diff(shown.to_numpy(), prepend=np.nan)),
                }
            )
        )

    return pd.concat(parts, ignore_index=True)


def step_profile(after_end: pd.DataFrame, after_start: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "po końcu odsypu": after_end.groupby("t_s")["step_kg"].mean(),
            "od początku odsypu": after_start.groupby("t_s")["step_kg"].mean(),
            "kontrola": control.groupby("t_s")["step_kg"].mean(),
        }
    )


def deviation_bands(table: pd.DataFrame) -> pd.DataFrame:
    quantiles = table.groupby("t_s")["deviation_kg"].quantile([0.1, 0.5, 0.9]).unstack()
    quantiles.columns = ["p10", "mediana", "p90"]
    return quantiles


def steady_threshold(control: pd.DataFrame) -> float:
    return float(_window_steps(control)["window_step_kg"].quantile(STEADY_QUANTILE))


def settling_times(after_end: pd.DataFrame, dispenses: pd.DataFrame, threshold: float) -> pd.DataFrame:
    windows = _window_steps(after_end)
    after = windows[windows["t_s"] >= 0]
    calm = after["window_step_kg"] <= threshold

    rows = []
    for number, group in after.groupby("number"):
        settled = group.loc[calm[group.index], "t_s"]
        rows.append({"number": number, "settled_s": settled.min() if not settled.empty else np.nan})

    table = pd.DataFrame(rows)
    table["start"] = dispenses["start"].to_numpy()
    table["end"] = dispenses["end"].to_numpy()
    table["free_after_s"] = dispenses["free_after_s"].to_numpy()
    return table[["start", "end", "free_after_s", "settled_s"]]


def step_figure(profile: pd.DataFrame, threshold: float, width: int) -> go.Figure:
    figure = go.Figure()
    for name, values in profile.items():
        figure.add_trace(go.Scatter(name=name, x=values.index, y=values, mode="lines+markers", marker={"size": 4}))

    figure.add_hline(y=threshold, line={"dash": "dot"}, annotation_text="próg spokoju")
    figure.update_layout(
        title="Średnia bezwzględna zmiana wskazania z sekundy na sekundę",
        xaxis_title="czas od chwili odniesienia [s]",
        yaxis_title="|zmiana| [kg]",
        width=width,
        height=FIGURE_HEIGHT,
    )
    return figure


def deviation_figure(after_end: pd.DataFrame, control: pd.DataFrame, width: int) -> go.Figure:
    figure = go.Figure()
    for name, table, color in (
        ("kontrola", control, CONTROL_BAND_COLOR),
        ("po końcu odsypu", after_end, BAND_COLOR),
    ):
        bands = deviation_bands(table)
        figure.add_trace(
            go.Scatter(
                x=[*bands.index, *bands.index[::-1]],
                y=[*bands["p90"], *bands["p10"][::-1]],
                fill="toself",
                fillcolor=color,
                line={"width": 0},
                name=f"{name}: 10.–90. percentyl",
                hoverinfo="skip",
            )
        )
        figure.add_trace(go.Scatter(x=bands.index, y=bands["mediana"], mode="lines", name=f"{name}: mediana"))

    figure.update_layout(
        title=f"Wskazanie względem mediany z ostatnich {REFERENCE_S} s okna",
        xaxis_title="czas od końca odsypu albo chwili kontrolnej [s]",
        yaxis_title="odchyłka [kg]",
        width=width,
        height=FIGURE_HEIGHT,
    )
    return figure


def overlay_figure(after_end: pd.DataFrame, width: int) -> go.Figure:
    figure = go.Figure()
    for _, group in after_end.groupby("number"):
        label = f"{group['moment'].iloc[0]:%Y-%m-%d %H:%M:%S}"
        figure.add_trace(
            go.Scatter(
                x=group["t_s"],
                y=group["deviation_kg"],
                mode="lines",
                line={"width": 1, "color": LINE_COLOR},
                name=label,
                hovertemplate=f"{label}<br>%{{x}} s: %{{y:.0f}} kg<extra></extra>",
                showlegend=False,
            )
        )

    figure.update_layout(
        title="Wskazania po końcu każdego odsypu (względem mediany z końca okna)",
        xaxis_title="czas od końca odsypu [s]",
        yaxis_title="odchyłka [kg]",
        width=width,
        height=FIGURE_HEIGHT,
    )
    return figure


def _window_steps(table: pd.DataFrame) -> pd.DataFrame:
    steps = table.groupby("number")["step_kg"].transform(
        lambda values: values.rolling(STEADY_WINDOW_S, min_periods=STEADY_WINDOW_S).mean().shift(-(STEADY_WINDOW_S - 1))
    )
    return table.assign(window_step_kg=steps).dropna(subset=["window_step_kg"])
