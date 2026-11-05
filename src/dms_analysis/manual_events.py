from dataclasses import dataclass

import pandas as pd

from dms_analysis.raw import TIMEZONE
from dms_analysis.signals import PIN_ROLES, REFILL_PIN, SignalEvents, Variant, detect_events


@dataclass(frozen=True)
class ManualEvent:
    pin: int
    start: pd.Timestamp
    end: pd.Timestamp


def _utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC").tz_convert(TIMEZONE)


MANUAL_EVENTS = {
    ("sqlite-2026-11-02", 32): [
        ManualEvent(REFILL_PIN, _utc("2026-10-30 13:14:10"), _utc("2026-10-30 13:36:49")),
    ],
}


def manual_events(source: str, session_id: int) -> list[ManualEvent]:
    return MANUAL_EVENTS.get((source, session_id), [])


def prepared_events(
    pins: pd.DataFrame, session_end: pd.Timestamp, variant: Variant, manual: list[ManualEvent]
) -> SignalEvents:
    replaced = {event.pin for event in manual}
    detected_pins = [int(pin) for pin in sorted(pins["pin"].unique()) if pin not in replaced]
    detected = detect_events(pins, session_end, pin_numbers=detected_pins, variant=variant)

    events = detected.events.assign(synthetic=False)
    if manual:
        events = pd.concat([events, _rows(manual)], ignore_index=True)

    return SignalEvents(
        events=events.sort_values(["pin", "start"]).reset_index(drop=True),
        too_short=detected.too_short.assign(synthetic=False),
        pulses=detected.pulses,
    )


def _rows(manual: list[ManualEvent]) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "role": [PIN_ROLES.get(event.pin, "") for event in manual],
            "pin": [event.pin for event in manual],
            "start": [event.start for event in manual],
            "end": [event.end for event in manual],
        }
    )
    duration = rows["end"] - rows["start"]

    return rows.assign(
        duration=duration,
        pulse_count=0,
        active_time=duration,
        active_share=1.0,
        open_start=False,
        open_end=False,
        synthetic=True,
    )
