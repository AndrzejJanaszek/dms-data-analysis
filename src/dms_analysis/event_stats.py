import math
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dms_analysis.prepared import PreparedSession

COLLISION_MARGIN = pd.Timedelta(seconds=8)
FIGURE_HEIGHT = 400
METHOD_ROW_HEIGHT = 250


@dataclass(frozen=True)
class WeightMethod:
    window: pd.Timedelta | None

    def description(self) -> str:
        if self.window is None:
            return "próbka najbliższa początkowi i końcowi sygnału"
        return f"mediana z {self.window.total_seconds():g} s przed początkiem i po końcu sygnału"


NEAREST = "najbliższa próbka"
METHODS = {
    "mediana 3 s": WeightMethod(pd.Timedelta(seconds=3)),
    NEAREST: WeightMethod(None),
}


def event_differences(session: PreparedSession) -> pd.DataFrame:
    events = session.events[~session.events["rejected"]].reset_index(drop=True)
    weight = session.data.weight.set_index("timestamp")["value"]

    base = pd.DataFrame(
        {
            "role": events["role"],
            "start": events["start"],
            "end": events["end"],
            "duration_s": events["duration"].dt.total_seconds(),
        }
    )
    base["collision"] = _collisions(base)

    tables = [_with_method(base, weight, name, method) for name, method in METHODS.items()]
    return pd.concat(tables, ignore_index=True)


def differences_by_method(table: pd.DataFrame) -> pd.DataFrame:
    wide = table.pivot_table(index=["role", "start"], columns="method", values="difference_kg", sort=False)
    return wide[list(METHODS)].reset_index()


def histogram(values: pd.Series, title: str, x_title: str, bin_size: float, width: int) -> go.Figure:
    figure = go.Figure(_bars(values, bin_size))
    figure.update_layout(
        title=f"{title} (n = {len(values)})",
        xaxis_title=x_title,
        yaxis_title="liczba zdarzeń",
        bargap=0,
        width=width,
        height=FIGURE_HEIGHT,
    )
    return figure


def method_histograms(
    table: pd.DataFrame, column: str, title: str, x_title: str, bin_size: float, width: int
) -> go.Figure:
    figure = make_subplots(rows=len(METHODS), cols=1, shared_xaxes=True, subplot_titles=list(METHODS))

    for row, name in enumerate(METHODS, start=1):
        figure.add_trace(_bars(table.loc[table["method"] == name, column], bin_size), row=row, col=1)
        figure.update_yaxes(title_text="liczba zdarzeń", row=row, col=1)

    count = len(table) // len(METHODS)
    figure.update_xaxes(title_text=x_title, showticklabels=True)
    figure.update_layout(
        title=f"{title} (n = {count})",
        bargap=0,
        showlegend=False,
        width=width,
        height=METHOD_ROW_HEIGHT * len(METHODS) + 100,
    )
    return figure


def _with_method(base: pd.DataFrame, weight: pd.Series, name: str, method: WeightMethod) -> pd.DataFrame:
    if method.window is None:
        before = _nearest(weight, base["start"])
        after = _nearest(weight, base["end"])
    else:
        window = method.window
        before = [_median(weight, start - window, start) for start in base["start"]]
        after = [_median(weight, end, end + window) for end in base["end"]]

    table = base.assign(method=name, before_kg=before, after_kg=after)
    table["difference_kg"] = table["after_kg"] - table["before_kg"]
    table["rate_kg_s"] = table["difference_kg"] / table["duration_s"]
    return table


def _nearest(weight: pd.Series, moments: pd.Series) -> list[float]:
    positions = weight.index.get_indexer(pd.DatetimeIndex(moments), method="nearest")
    return weight.iloc[positions].astype(float).tolist()


def _median(weight: pd.Series, low: pd.Timestamp, high: pd.Timestamp) -> float:
    return float(weight[(weight.index >= low) & (weight.index < high)].median())


def _collisions(table: pd.DataFrame) -> list[bool]:
    low = table["start"] - COLLISION_MARGIN
    high = table["end"] + COLLISION_MARGIN
    collisions = []

    for row in table.itertuples():
        others = table["role"] != row.role
        overlapping = others & (low < high[row.Index]) & (high > low[row.Index])
        collisions.append(bool(overlapping.any()))

    return collisions


def _bars(values: pd.Series, bin_size: float) -> go.Histogram:
    start = math.floor(values.min() / bin_size) * bin_size - bin_size / 2
    bins = {"start": start, "size": bin_size}
    return go.Histogram(x=values, xbins=bins, marker={"line": {"width": 1, "color": "white"}})
