from dataclasses import dataclass

import pandas as pd

ACTIVE_LEVEL = 0
DISPENSE_PIN = 5
REFILL_PIN = 4
PIN_ROLES = {DISPENSE_PIN: "odsyp", REFILL_PIN: "zasyp"}

MAX_GAP = pd.Timedelta(seconds=1)
MIN_DURATION = pd.Timedelta(seconds=1)


@dataclass(frozen=True)
class SignalEvents:
    events: pd.DataFrame
    too_short: pd.DataFrame
    pulses: pd.DataFrame


def detect_events(
    pins: pd.DataFrame,
    session_end: pd.Timestamp,
    pin_numbers: list[int] | None = None,
    max_gap: pd.Timedelta = MAX_GAP,
    min_duration: pd.Timedelta = MIN_DURATION,
    active_level: int = ACTIVE_LEVEL,
) -> SignalEvents:
    selected = sorted(pins["pin"].unique()) if pin_numbers is None else pin_numbers
    pulses = []
    merged = []

    for pin in selected:
        changes = pins[pins["pin"] == pin]
        pin_pulses = active_pulses(changes, session_end, active_level).assign(pin=pin)
        pin_events = merge_pulses(pin_pulses, max_gap).assign(pin=pin)
        pin_events["open_start"] = pin_events["start"] == changes["timestamp"].iloc[0]
        pin_events["open_end"] = pin_events["end"] == session_end
        pulses.append(pin_pulses)
        merged.append(pin_events)

    pulses_table = pd.concat(pulses, ignore_index=True)
    events = _with_roles(pd.concat(merged, ignore_index=True))
    long_enough = events["duration"] >= min_duration

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


def merge_pulses(pulses: pd.DataFrame, max_gap: pd.Timedelta = MAX_GAP) -> pd.DataFrame:
    gap_before = pulses["start"] - pulses["end"].shift()
    event_number = (~(gap_before < max_gap)).cumsum()
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
