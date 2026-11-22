import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display
from pandas.io.formats.style import Styler
from plotly.subplots import make_subplots

from dms_analysis import style
from dms_analysis.data import check
from dms_analysis.data.review import grafana_link
from dms_analysis.errors import store
from dms_analysis.errors.balance import CONTROL, GROUPS, OUTSIDE_PERIOD, VARIANTS, WITH_REFILL, WITHOUT_REFILL
from dms_analysis.errors.stats import BOOTSTRAP_SAMPLES, CONFIDENCE, SEED

GROUP_COLORS = {CONTROL: style.AQUA, WITHOUT_REFILL: style.BLUE, WITH_REFILL: style.ORANGE}
MIN_GROUP_DAYS = 10
ENVELOPE_BINS = 3000
BROWSER_MARGIN = pd.Timedelta(minutes=30)
REAL = "real"
TIME_NAMES = {"real": "rzeczywisty", "estimated": "szacowany ±2 h", "unknown": "zegara"}


def load_balances(periods: list[str]) -> pd.DataFrame:
    frames = []
    for full_id in periods:
        tables = store.load_balance(full_id)
        frames.append(tables["balance"].assign(period=full_id, time=tables["run"]["time"].iloc[0]))
    return pd.concat(frames, ignore_index=True)


def valid(balances: pd.DataFrame, variant: str) -> pd.DataFrame:
    return balances[(balances["variant"] == variant) & (balances["valid"] == 1)].sort_values("date")


def counts_table(balances: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        chosen = valid(balances, variant)
        for time in (REAL, "estimated"):
            subset = chosen[chosen["time"] == time]
            counts = subset["group"].value_counts()
            rows.append(
                {
                    "bilans": variant,
                    "czas": TIME_NAMES[time],
                    **{group: int(counts.get(group, 0)) for group in GROUPS},
                    "z zakłóceniem": int(subset["disturbance"].sum()),
                }
            )
    return pd.DataFrame(rows).set_index(["bilans", "czas"])


def invalid_table(balances: pd.DataFrame) -> pd.DataFrame:
    chosen = balances[(balances["valid"] == 0) & (balances["reason"] != OUTSIDE_PERIOD)]
    table = chosen[["variant", "period", "date", "reason"]].rename(
        columns={"variant": "bilans", "period": "okres", "date": "doba", "reason": "powód"}
    )
    outside = int(((balances["valid"] == 0) & (balances["reason"] == OUTSIDE_PERIOD)).sum())
    print(f"doby poza okresem albo z luką (pierwsza i ostatnia doba okresu itp.): {outside}")
    return table.reset_index(drop=True)


def components_table(balances: pd.DataFrame, variant: str) -> Styler:
    chosen = valid(balances, variant)
    table = pd.DataFrame(
        {
            "okres": chosen["period"].str.split("/").str[1].str[:4],
            "doba": chosen["date"],
            "czas": chosen["time"].map(TIME_NAMES),
            "grupa": chosen["group"],
            "W początek": _whole(chosen["w_start_kg"]),
            "W koniec": _whole(chosen["w_end_kg"]),
            "ΔW": _whole(chosen["change_kg"]),
            "zasypy": chosen["refills"].astype(int),
            "Σ zasypów": _whole(chosen["refill_kg"]),
            "odsypy": chosen["dispenses"].astype(int),
            "Σ odsypów": _whole(chosen["dispense_kg"]),
            "odsypy w zasypie": chosen["dispenses_in_refill"].astype(int),
            "z przeglądu": chosen["review_accepted"].astype(int),
            "zakłócenie": chosen["disturbance"].map({0: "", 1: "tak"}),
            "doba [h]": _whole(chosen["hours"]),
            "R": _whole(chosen["r_kg"]),
        }
    )
    limit = float(table["R"].abs().max()) or 1.0
    return (
        table.style.format(thousands=" ")
        .map(lambda value: style.diverging_css(value, limit), subset=["R"])
        .hide(axis="index")
        .set_caption(
            f"Bilans {variant}: R = ΔW − Σ zasypów + Σ odsypów [kg] (odsypy w trakcie zasypu w bloku zasypu, "
            "nie liczone osobno; „z przeglądu” = odsypy odrzucone przez regułę, przyjęte w przeglądzie)"
        )
    )


def _whole(values: pd.Series) -> pd.Series:
    return values.round().astype(int)


def points_figure(balances: pd.DataFrame) -> go.Figure:
    figure = make_subplots(
        rows=len(VARIANTS), cols=1, vertical_spacing=0.14, subplot_titles=[f"bilans {variant}" for variant in VARIANTS]
    )
    for row, variant in enumerate(VARIANTS, start=1):
        chosen = valid(balances, variant)
        for group in GROUPS:
            for time in (REAL, "estimated"):
                subset = chosen[(chosen["group"] == group) & (chosen["time"] == time)]
                if subset.empty:
                    continue
                figure.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(subset["date"]),
                        y=subset["r_kg"],
                        mode="markers+text",
                        text=pd.to_datetime(subset["date"]).dt.strftime("%d.%m"),
                        textposition="top center",
                        textfont={"size": 10, "color": style.TEXT_SECONDARY},
                        marker={
                            "size": 11,
                            "color": GROUP_COLORS[group] if time == REAL else style.SURFACE,
                            "line": {"color": GROUP_COLORS[group], "width": 2},
                            "symbol": np.where(subset["disturbance"] == 1, "x", "circle"),
                        },
                        name=group if time == REAL else f"{group} (czas szacowany)",
                        legendgroup=f"{group}-{time}",
                        showlegend=row == 1,
                        customdata=np.column_stack([subset["dispenses"], subset["refills"], subset["period"]]),
                        hovertemplate=(
                            "%{x|%Y-%m-%d}: R = %{y:.0f} kg<br>odsypy %{customdata[0]}, zasypy %{customdata[1]}"
                            "<br>%{customdata[2]}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=1,
                )
        figure.add_hline(y=0, line={"color": style.TEXT_SECONDARY, "width": 1}, row=row, col=1)
        figure.update_yaxes(title_text="R [kg]", row=row, col=1)
    figure.update_layout(
        title=(
            "Bilans masy doby: R na każdą dobę · kolor = grupa, puste kółko = czas szacowany, × = zakłócenie w dobie"
        ),
        height=820,
    )
    return figure


def mean_interval(values: np.ndarray, samples: int = BOOTSTRAP_SAMPLES) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    means = values[rng.integers(0, len(values), (samples, len(values)))].mean(axis=1)
    tail = (1 - CONFIDENCE) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(low), float(high)


def groups_view(balances: pd.DataFrame) -> go.Figure | None:
    rows = []
    for variant in VARIANTS:
        chosen = valid(balances, variant)
        chosen = chosen[chosen["time"] == REAL]
        for group in GROUPS:
            values = chosen[chosen["group"] == group]["r_kg"].to_numpy(dtype=float)
            rows.append((variant, group, values))

    for variant, group, values in rows:
        status = "pudełko i przedział" if len(values) >= MIN_GROUP_DAYS else f"za mało dób (< {MIN_GROUP_DAYS})"
        print(f"{variant}, {group}: {len(values)} dób z czasem rzeczywistym — {status}")

    shown = [(variant, group, values) for variant, group, values in rows if len(values) >= MIN_GROUP_DAYS]
    if not shown:
        return None

    figure = go.Figure()
    for variant, group, values in shown:
        low, high = mean_interval(values)
        label = f"{variant}<br>{group} (n = {len(values)})"
        figure.add_trace(
            go.Box(
                y=values,
                x=[label] * len(values),
                name=label,
                boxpoints="all",
                jitter=0.4,
                pointpos=0,
                marker={"color": GROUP_COLORS[group]},
                line={"color": GROUP_COLORS[group]},
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[label],
                y=[values.mean()],
                mode="markers",
                marker={"symbol": "diamond", "size": 12, "color": style.TEXT},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [high - values.mean()],
                    "arrayminus": [values.mean() - low],
                },
                name="średnia z przedziałem 95 % (bootstrap po dobach)",
                showlegend=False,
                hovertemplate=f"średnia %{{y:.0f}} kg [{low:.0f}…{high:.0f}]<extra></extra>",
            )
        )
    figure.add_hline(y=0, line={"color": style.TEXT_SECONDARY, "width": 1})
    figure.update_layout(
        title="Grupy z co najmniej 10 dobami (czas rzeczywisty) · romb: średnia R z przedziałem 95 %",
        yaxis_title="R [kg]",
        height=480,
    )
    return figure


class DayBrowser:
    def __init__(self, balances: pd.DataFrame) -> None:
        self._balances = balances
        self._periods: dict[str, check.PreparedPeriod] = {}
        self._variant = widgets.ToggleButtons(options=list(VARIANTS), description="bilans")
        self._day = widgets.Dropdown(description="doba", layout={"width": "520px"})
        self._info = widgets.HTML()
        self._output = widgets.Output()

        self._variant.observe(lambda _: self._options(), names="value")
        self._day.observe(lambda _: self._show(), names="value")
        self._options()

    def widget(self) -> widgets.VBox:
        return widgets.VBox([self._variant, self._day, self._info, self._output])

    def _options(self) -> None:
        chosen = valid(self._balances, self._variant.value)
        self._day.options = [
            (f"{row.date} · {row.group} · R = {row.r_kg:+.0f} kg · {row.period}", index)
            for index, row in zip(chosen.index, chosen.itertuples(), strict=True)
        ]

    def _period(self, full_id: str) -> check.PreparedPeriod:
        if full_id not in self._periods:
            self._periods = {full_id: check.load_period(full_id)}
        return self._periods[full_id]

    def _show(self) -> None:
        if self._day.value is None:
            return
        row = self._balances.loc[self._day.value]
        period = self._period(row["period"])
        first = _local(row["start_window_start"]), _local(row["start_window_end"])
        last = _local(row["end_window_start"]), _local(row["end_window_end"])
        link = grafana_link(row["period"], pd.Series({"start": first[0], "end": last[1]}))
        self._info.value = (
            f"okna odniesienia: {first[0]:%d.%m %H:%M}–{first[1]:%H:%M} i {last[0]:%d.%m %H:%M}–{last[1]:%H:%M} "
            f"czasu lokalnego (UTC: {first[0].tz_convert('UTC'):%d.%m %H:%M}–{first[1].tz_convert('UTC'):%H:%M} i "
            f"{last[0].tz_convert('UTC'):%d.%m %H:%M}–{last[1].tz_convert('UTC'):%H:%M}) · "
            f"<a href='{link}' target='_blank'>Grafana ↗</a>"
        )
        figure = day_figure(period, row, first, last)
        with self._output:
            self._output.clear_output(wait=True)
            figure.show()


def _local(seconds: float) -> pd.Timestamp:
    return pd.Timestamp(seconds, unit="s", tz="UTC").tz_convert(check.TIMEZONE)


def day_figure(
    period: check.PreparedPeriod,
    row: pd.Series,
    first: tuple[pd.Timestamp, pd.Timestamp],
    last: tuple[pd.Timestamp, pd.Timestamp],
) -> go.Figure:
    low, high = first[0] - BROWSER_MARGIN, last[1] + BROWSER_MARGIN
    weight = period.weight[(period.weight["timestamp"] >= low) & (period.weight["timestamp"] <= high)]
    reduced = check.envelope(weight, "value", ENVELOPE_BINS)

    figure = go.Figure(
        go.Scatter(
            x=reduced["timestamp"].dt.tz_localize(None),
            y=reduced["value"],
            mode="lines",
            line={"width": 1, "color": style.WEIGHT},
            name="wskazanie",
        )
    )
    events = period.events[
        (period.events["end"] >= low) & (period.events["start"] <= high) & (period.events["rejected"] == 0)
    ]
    for event in events.itertuples():
        color = check.MANUAL_COLOR if event.role == "zasyp" else style.SIGNAL_BAND
        figure.add_vrect(
            x0=event.start.tz_localize(None), x1=event.end.tz_localize(None), fillcolor=color, line_width=0
        )
    for (start, end), value, name in ((first, row["w_start_kg"], "W początek"), (last, row["w_end_kg"], "W koniec")):
        figure.add_vrect(
            x0=start.tz_localize(None), x1=end.tz_localize(None), fillcolor=style.WINDOW_BAND, line_width=0
        )
        figure.add_trace(
            go.Scatter(
                x=[start.tz_localize(None), end.tz_localize(None)],
                y=[value, value],
                mode="lines",
                line={"color": style.RED, "width": 3},
                name=f"{name}: {value:.0f} kg",
            )
        )
    figure.update_layout(
        title=(
            f"{row['date']} · bilans {row['variant']} · {row['group']} · R = {row['r_kg']:+.0f} kg "
            f"(ΔW {row['change_kg']:+.0f}, zasypy {row['refill_kg']:.0f}, odsypy {row['dispense_kg']:.0f})<br>"
            "<span style='font-size:12px'>niebieskie pasma: okna odniesienia (czerwona kreska = mediana), "
            "żółte: zasypy, szare: odsypy; czas lokalny</span>"
        ),
        yaxis_title="wskazanie [kg]",
        height=520,
    )
    return figure


def show_days(balances: pd.DataFrame) -> None:
    display(DayBrowser(balances).widget())
