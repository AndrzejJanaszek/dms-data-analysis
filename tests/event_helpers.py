import numpy as np
import pandas as pd

from dms_analysis.events.values import PeriodData, Weight


def weight_from(values: list[float], start: float = 0.0) -> Weight:
    return Weight(start + np.arange(len(values), dtype=float), np.array(values, dtype=float))


def events_frame(rows: list[tuple[str, float, float, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["role", "start", "end", "rejected", "synthetic"])


def period_data(
    weight: Weight,
    events: pd.DataFrame,
    issues: pd.DataFrame | None = None,
    parts: pd.DataFrame | None = None,
) -> PeriodData:
    return PeriodData(
        full_id="test-1/0001-2026-11-02",
        number=1,
        time="real",
        division_kg=20.0,
        has_temperatures=True,
        weight=weight,
        events=events,
        issues=issues if issues is not None else pd.DataFrame(columns=["channel", "kind", "start", "end", "note"]),
        parts=parts if parts is not None else pd.DataFrame({"part": [1], "start": [weight.seconds[0]], "end": [weight.seconds[-1]]}),
        prepared_at="2026-11-11T00:00:00+00:00",
        catalog_sha256="0" * 64,
    )
