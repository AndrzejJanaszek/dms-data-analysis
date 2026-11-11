from pathlib import Path
from urllib.parse import urlencode

import ipywidgets as widgets
import pandas as pd
from IPython.display import display

from dms_analysis.data import check
from dms_analysis.data.catalog import Catalog
from dms_analysis.data.paths import DISPENSE_MARKS

DISPENSE = "odsyp"
DEFAULT_MARGIN_S = 60.0
GRAFANA_MARGIN = pd.Timedelta(minutes=10)
GRAFANA_DASHBOARD = "http://localhost:3000/d/dms-periods-overview/okresy"

MARK_COLUMNS = ["period", "start", "end", "mark"]
BAD = "zły"
GOOD = "dobry"
MARK_BUTTONS = {BAD: "✗ zły", GOOD: "✓ dobry"}
MARK_STYLES = {BAD: "danger", GOOD: "success"}
MARK_COLORS = {BAD: "#c0392b", GOOD: "#2d7a46"}
SHOW_ALL = "wszystkie"
SHOW_SUSPICIOUS = "tylko podejrzane"
SHOW_BAD = "tylko złe"


def dispense_periods(catalog: Catalog) -> list[str]:
    return [full_id for full_id, period in catalog.periods.items() if period.channels["dispense"] in ("ok", "flickering")]


def review_order(period: check.PreparedPeriod) -> pd.DataFrame:
    dispenses = period.events[(period.events["role"] == DISPENSE) & (period.events["synthetic"] == 0)]
    events = check.events_weight_changes(period, dispenses)

    reasons = events.apply(_suspicion, axis=1)
    events = events.assign(reason=reasons, suspicious=reasons != "")
    return events.sort_values(["suspicious", "start"], ascending=[False, True]).reset_index(drop=True)


def _suspicion(event: pd.Series) -> str:
    reasons = []
    if event["rejected"]:
        reasons.append("odrzucony")
    if pd.isna(event["change"]):
        reasons.append("brak wskazań wokół")
    elif event["change"] >= 0:
        reasons.append("bez spadku wskazań")
    return ", ".join(reasons)


def load_marks(path: Path = DISPENSE_MARKS) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=MARK_COLUMNS)

    marks = pd.read_csv(path, dtype=str)
    if "mark" not in marks.columns:
        marks = marks.assign(mark=BAD)
    return marks[MARK_COLUMNS]


def save_marks(marks: pd.DataFrame, path: Path = DISPENSE_MARKS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    marks.sort_values(MARK_COLUMNS).to_csv(path, index=False)


def _matches(marks: pd.DataFrame, full_id: str, event: pd.Series) -> pd.Series:
    return (marks["period"] == full_id) & (marks["start"] == event["start"].isoformat())


def event_mark(marks: pd.DataFrame, full_id: str, event: pd.Series) -> str | None:
    found = marks[_matches(marks, full_id, event)]
    return None if found.empty else str(found["mark"].iloc[0])


def toggle_mark(marks: pd.DataFrame, full_id: str, event: pd.Series, mark: str) -> pd.DataFrame:
    previous = event_mark(marks, full_id, event)
    remaining = marks[~_matches(marks, full_id, event)].reset_index(drop=True)
    if previous == mark:
        return remaining

    row = {"period": full_id, "start": event["start"].isoformat(), "end": event["end"].isoformat(), "mark": mark}
    return pd.concat([remaining, pd.DataFrame([row])], ignore_index=True)


def grafana_link(full_id: str, event: pd.Series) -> str:
    start = event["start"] - GRAFANA_MARGIN
    end = event["end"] + GRAFANA_MARGIN
    query = urlencode({"var-period": full_id, "from": int(start.timestamp() * 1000), "to": int(end.timestamp() * 1000)})
    return f"{GRAFANA_DASHBOARD}?{query}"


class DispenseBrowser:
    def __init__(self, periods: list[str], period: str | None = None, view: str = SHOW_ALL) -> None:
        self._period: check.PreparedPeriod | None = None
        self._events = pd.DataFrame()
        self._position = 0
        self._marks = load_marks()

        self._all_events = pd.DataFrame()
        self._choice = widgets.Dropdown(
            options=periods, value=period or periods[0], description="okres", layout={"width": "480px"}
        )
        self._filter = widgets.ToggleButtons(options=[SHOW_ALL, SHOW_SUSPICIOUS, SHOW_BAD], value=view, description="pokaż")
        self._margin = widgets.FloatText(value=DEFAULT_MARGIN_S, description="margines [s]", layout={"width": "180px"})
        self._previous = widgets.Button(description="◀ poprzedni")
        self._next = widgets.Button(description="następny ▶")
        self._number = widgets.BoundedIntText(min=1, max=1, description="nr", layout={"width": "150px"})
        self._count = widgets.Label()
        self._mark_buttons = {
            mark: widgets.Button(description=label, layout={"width": "140px"}) for mark, label in MARK_BUTTONS.items()
        }
        self._marked_count = widgets.Label()
        self._summary = widgets.HTML()
        self._info = widgets.HTML()
        self._output = widgets.Output()

        self._choice.observe(lambda _: self._load(), names="value")
        self._filter.observe(lambda _: self._select(), names="value")
        self._margin.observe(lambda _: self._show(), names="value")
        self._number.observe(lambda change: self._go_to(change["new"] - 1), names="value")
        self._previous.on_click(lambda _: self._go_to(self._position - 1))
        self._next.on_click(lambda _: self._go_to(self._position + 1))
        for mark, button in self._mark_buttons.items():
            button.on_click(lambda _, mark=mark: self._toggle_mark(mark))

        self._load()

    def widget(self) -> widgets.VBox:
        return widgets.VBox(
            [
                widgets.HBox([self._choice, self._margin]),
                self._filter,
                self._summary,
                widgets.HBox([self._previous, self._next, self._number, self._count, *self._mark_buttons.values()]),
                self._marked_count,
                self._info,
                self._output,
            ]
        )

    def _load(self) -> None:
        self._period = check.load_period(self._choice.value)
        self._all_events = review_order(self._period)
        self._select()

    def _select(self) -> None:
        events = self._all_events
        if self._filter.value == SHOW_SUSPICIOUS:
            events = events[events["suspicious"]].sort_values("start").reset_index(drop=True)
        if self._filter.value == SHOW_BAD:
            marked = [event_mark(self._marks, self._choice.value, event) == BAD for _, event in events.iterrows()]
            events = events[marked].sort_values("start").reset_index(drop=True)
        self._events = events
        self._summary.value = _view_summary(self._filter.value, self._all_events, events)
        self._count.value = f"z {len(self._events)}"
        self._number.max = max(len(self._events), 1)
        self._position = 0
        self._number.value = 1
        self._show()

    def _toggle_mark(self, mark: str) -> None:
        if self._events.empty:
            return

        event = self._events.iloc[self._position]
        self._marks = toggle_mark(self._marks, self._choice.value, event, mark)
        save_marks(self._marks)

        if event_mark(self._marks, self._choice.value, event) and self._position + 1 < len(self._events):
            self._go_to(self._position + 1)
        else:
            self._show()

    def _go_to(self, position: int) -> None:
        if not 0 <= position < len(self._events) or position == self._position:
            return
        self._position = position
        self._number.value = position + 1
        self._show()

    def _show(self) -> None:
        period = self._period
        if period is None or self._events.empty:
            self._info.value = "<b>brak odsypów w tym okresie</b>"
            self._output.clear_output()
            return

        full_id = self._choice.value
        event = self._events.iloc[self._position]
        mark = event_mark(self._marks, full_id, event)
        self._info.value = _description(event, self._position, len(self._events), mark, grafana_link(full_id, event))
        self._marked_count.value = _marks_summary(self._marks, full_id)
        for button_mark, button in self._mark_buttons.items():
            button.button_style = MARK_STYLES[button_mark] if mark == button_mark else ""

        margin = pd.Timedelta(seconds=self._margin.value)
        figure = check.event_figure(
            period, event["start"] - margin, event["end"] + margin, period.title, (event["start"], event["end"])
        )

        with self._output:
            self._output.clear_output(wait=True)
            figure.show()


def _view_summary(view: str, all_events: pd.DataFrame, events: pd.DataFrame) -> str:
    if view == SHOW_SUSPICIOUS:
        rejected = int(events["rejected"].sum())
        return (
            f"<b>{len(events)} podejrzanych odsypów</b> ({rejected} odrzuconych przez reguły, "
            f"{len(events) - rejected} przyjętych bez spadku wskazań), po kolei w czasie; "
            f"dobre oznacz „✓ dobry”, pozostałe są traktowane jako złe"
        )
    if view == SHOW_BAD:
        return (
            f"<b>{len(events)} odsypów oznaczonych jako złe</b>, po kolei w czasie; "
            f"„✗ zły” cofa oznaczenie (odsyp zostaje na liście do zmiany filtra)"
        )
    suspicious = int(all_events["suspicious"].sum())
    return (
        f"{len(all_events)} odsypów, w tym <b>{suspicious} podejrzanych</b> na początku listy "
        f"(odrzucone albo bez spadku wskazań), dalej pozostałe po kolei w czasie"
    )


def _marks_summary(marks: pd.DataFrame, full_id: str) -> str:
    counts = marks[marks["period"] == full_id]["mark"].value_counts()
    parts = ", ".join(f"{mark}: {int(counts.get(mark, 0))}" for mark in MARK_BUTTONS)
    return f"oznaczone w tym okresie — {parts} (ponowne kliknięcie cofa; zapis: data/review/{DISPENSE_MARKS.name})"


def _description(event: pd.Series, position: int, count: int, mark: str | None, link: str) -> str:
    change = "—" if pd.isna(event["change"]) else f"{event['change']:+.0f} kg"
    status = "odrzucony" if event["rejected"] else "przyjęty"
    reason = f" · <span style='color:{check.CURRENT_EVENT_COLOR}'><b>podejrzany: {event['reason']}</b></span>"
    marked = f"<span style='color:{MARK_COLORS[mark]}'><b>oznaczony: {mark}</b></span> · " if mark else ""
    return (
        f"<b>{position + 1}/{count}</b> · {event['start']:%Y-%m-%d %H:%M:%S} – {event['end']:%H:%M:%S}"
        f" · {event['duration']:.1f} s · {event['pulse_count']} impulsów · {status}"
        f" · zmiana mediany 3 s: {change}"
        + (reason if event["suspicious"] else "")
        + f"<br>{marked}<a href='{link}' target='_blank'>otwórz w Grafanie (±10 min) ↗</a>"
    )


def show(catalog: Catalog, period: str | None = None, view: str = SHOW_ALL) -> None:
    display(DispenseBrowser(dispense_periods(catalog), period, view).widget())
