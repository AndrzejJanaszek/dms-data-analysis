from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ACTIVE_LEVEL = 0
DISPENSE_PIN = 5
REFILL_PIN = 4
PIN_ROLES = {DISPENSE_PIN: "odsyp", REFILL_PIN: "zasyp"}


PIECE_GAP = pd.Timedelta(seconds=1)


@dataclass(frozen=True)
class EventRules:
    max_gap: pd.Timedelta
    min_duration: pd.Timedelta
    length_factor: float | None = None
    piece_gap: pd.Timedelta = PIECE_GAP

    def __str__(self) -> str:
        if self.length_factor is None:
            merging = f"łączenie przy przerwie < {_seconds(self.max_gap)} s"
        else:
            merging = (
                f"kawałki z przerwą < {_seconds(self.piece_gap)} s, łączenie kawałków przy przerwie"
                f" < min({_seconds(self.max_gap)} s, {self.length_factor:g} × dłuższy z sąsiadów)"
            )
        return f"{merging}, odrzucenie zdarzeń < {_seconds(self.min_duration)} s"


@dataclass(frozen=True)
class Variant:
    description: str
    default: EventRules
    pins: dict[int, EventRules] = field(default_factory=dict)

    def rules(self, pin: int) -> EventRules:
        return self.pins.get(pin, self.default)


def _rules(max_gap: float, min_duration: float, length_factor: float | None = None) -> EventRules:
    return EventRules(pd.Timedelta(seconds=max_gap), pd.Timedelta(seconds=min_duration), length_factor)


VARIANTS = {
    "v1": Variant("przerwa 1 s, min. 1 s", _rules(1, 1)),
    "v2": Variant("przerwa 5 s, min. 1 s", _rules(5, 1)),
    "v3": Variant("odsyp: przerwa 8 s, min. 10 s", _rules(5, 1), {DISPENSE_PIN: _rules(8, 10)}),
    "v4": Variant(
        "odsyp: przerwa zależna od długości (k = 1, maks. 8 s), min. 10 s",
        _rules(5, 1),
        {DISPENSE_PIN: _rules(8, 10, length_factor=1)},
    ),
}
CURRENT_VARIANT = "v4"


@dataclass(frozen=True)
class SignalEvents:
    events: pd.DataFrame
    too_short: pd.DataFrame
    pulses: pd.DataFrame


def detect_events(
    pins: pd.DataFrame,
    session_end: pd.Timestamp,
    pin_numbers: list[int] | None = None,
    variant: Variant = VARIANTS[CURRENT_VARIANT],
    active_level: int = ACTIVE_LEVEL,
) -> SignalEvents:
    selected = sorted(pins["pin"].unique()) if pin_numbers is None else pin_numbers
    pulses = []
    merged = []

    for pin in selected:
        changes = pins[pins["pin"] == pin]
        rules = variant.rules(pin)
        pin_pulses = active_pulses(changes, session_end, active_level).assign(pin=pin)
        pin_events = merge_pulses(pin_pulses, rules).assign(pin=pin)
        pin_events["open_start"] = pin_events["start"] == changes["timestamp"].iloc[0]
        pin_events["open_end"] = pin_events["end"] == session_end
        pin_events["long_enough"] = pin_events["duration"] >= rules.min_duration
        pulses.append(pin_pulses)
        merged.append(pin_events)

    pulses_table = pd.concat(pulses, ignore_index=True)
    events = pd.concat(merged, ignore_index=True)
    long_enough = events.pop("long_enough")
    events = _with_roles(events)

    return SignalEvents(
        events=events[long_enough].reset_index(drop=True),
        too_short=events[~long_enough].reset_index(drop=True),
        pulses=pulses_table,
    )


def active_pulses(
    changes: pd.DataFrame, session_end: pd.Timestamp, active_level: int = ACTIVE_LEVEL
) -> pd.DataFrame:
    states = changes[changes["value"].diff() != 0]
    following = states["timestamp"].shift(-1).fillna(session_end)
    active = states["value"] == active_level

    return pd.DataFrame(
        {"start": states["timestamp"][active], "end": following[active]}
    ).reset_index(drop=True)


def merge_pulses(pulses: pd.DataFrame, rules: EventRules) -> pd.DataFrame:
    if rules.length_factor is None:
        return _aggregate(pulses, _numbered(pulses, rules.max_gap))

    piece_number = _numbered(pulses, rules.piece_gap)
    groups = _length_aware_groups(_aggregate(pulses, piece_number), rules)
    return _aggregate(pulses, pd.Series(groups[piece_number.to_numpy() - 1], index=pulses.index))


def _numbered(pulses: pd.DataFrame, max_gap: pd.Timedelta) -> pd.Series:
    gap_before = pulses["start"] - pulses["end"].shift()
    return (~(gap_before < max_gap)).cumsum()


def _length_aware_groups(pieces: pd.DataFrame, rules: EventRules) -> np.ndarray:
    groups = np.zeros(len(pieces), dtype=int)
    group = 0
    event_start = event_end = None

    for position, piece in enumerate(pieces.itertuples()):
        if event_end is None or not _joins(piece.start, piece.end, event_start, event_end, rules):
            group += 1
            event_start, event_end = piece.start, piece.end
        event_end = max(event_end, piece.end)
        groups[position] = group

    return groups


def _joins(
    start: pd.Timestamp, end: pd.Timestamp, event_start: pd.Timestamp, event_end: pd.Timestamp, rules: EventRules
) -> bool:
    longer = max(event_end - event_start, end - start)
    limit = min(rules.max_gap, longer * rules.length_factor)
    return start - event_end < limit


def _aggregate(pulses: pd.DataFrame, event_number: pd.Series) -> pd.DataFrame:
    lengths = pulses.assign(active_time=pulses["end"] - pulses["start"])

    events = lengths.groupby(event_number).agg(
        start=("start", "min"),
        end=("end", "max"),
        pulse_count=("start", "size"),
        active_time=("active_time", "sum"),
    )
    events["duration"] = events["end"] - events["start"]
    events["active_share"] = events["active_time"] / events["duration"]
    return events.reset_index(drop=True)


def cleaned_changes(
    events: pd.DataFrame, pins: pd.DataFrame, active_level: int = ACTIVE_LEVEL
) -> pd.DataFrame:
    inactive_level = 1 - active_level
    first_records = pins.groupby("pin", as_index=False)["timestamp"].min()
    pin_numbers = events["pin"].unique()

    initial = first_records[first_records["pin"].isin(pin_numbers)].assign(value=inactive_level)
    starts = events[["pin", "start"]].rename(columns={"start": "timestamp"}).assign(value=active_level)
    ends = events.loc[~events["open_end"], ["pin", "end"]].rename(columns={"end": "timestamp"})

    changes = pd.concat([initial, starts, ends.assign(value=inactive_level)], ignore_index=True)
    changes = changes.drop_duplicates(subset=["pin", "timestamp"], keep="last")
    return changes.sort_values(["pin", "timestamp"]).reset_index(drop=True)[["timestamp", "pin", "value"]]


def summary(signal_events: SignalEvents) -> pd.DataFrame:
    counts = {
        "impulsy": signal_events.pulses.groupby("pin").size(),
        "zdarzenia": signal_events.events.groupby("pin").size(),
        "odrzucone krótkie": signal_events.too_short.groupby("pin").size(),
    }
    table = pd.DataFrame(counts).fillna(0).astype(int)
    table.insert(0, "rola", [PIN_ROLES.get(int(pin), "") for pin in table.index])
    return table


def _with_roles(events: pd.DataFrame) -> pd.DataFrame:
    events.insert(0, "role", events["pin"].map(PIN_ROLES).fillna(""))
    columns = ["role", "pin", "start", "end", "duration", "pulse_count", "active_time", "active_share"]
    return events[[*columns, "open_start", "open_end"]]


def events_table(events: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rola": events["role"],
            "pin": events["pin"],
            "początek": events["start"].dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-4],
            "koniec": events["end"].dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-4],
            "czas [s]": events["duration"].dt.total_seconds().round(2),
            "impulsy": events["pulse_count"],
            "udział aktywnego": events["active_share"].round(2),
            "otwarte na początku": events["open_start"],
            "otwarte na końcu": events["open_end"],
        }
    )


def _seconds(value: pd.Timedelta) -> str:
    return f"{value.total_seconds():g}"
