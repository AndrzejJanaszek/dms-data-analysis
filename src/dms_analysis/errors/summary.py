import numpy as np
import pandas as pd

from dms_analysis.errors import stats
from dms_analysis.errors.fragments import DAY_PARTS, part_of_hour
from dms_analysis.events.values import METHODS

REAL_TIME_SCOPE = "czas rzeczywisty"
GROUP_KEYS = ["duration_s", "delay_s", "method", "part"]
STATS_COLUMNS = ["scope", *GROUP_KEYS, *stats.SUMMARY_COLUMNS]


def day_counts(counts: pd.DataFrame) -> pd.DataFrame:
    frame = counts.assign(part=counts["hour"].map(part_of_hour), unit=counts["date"])
    return frame.groupby(["unit", *GROUP_KEYS, "error_kg"], as_index=False)["starts"].sum()


def scope_stats(scope: str, scope_index: int, counts: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for key, group in day_counts(counts).groupby(GROUP_KEYS, sort=True):
        duration_s, delay_s, method, part = key
        rng = np.random.default_rng(
            [stats.SEED, scope_index, int(duration_s), int(delay_s * 10), METHODS.index(method), DAY_PARTS.index(part)]
        )
        table = stats.summary(group, rng)
        frames.append(table.assign(scope=scope, duration_s=duration_s, delay_s=delay_s, method=method, part=part))
    return pd.concat(frames, ignore_index=True)[STATS_COLUMNS]


def site_stats(period_counts: dict[str, pd.DataFrame], real_time: list[str]) -> pd.DataFrame:
    frames = [scope_stats(full_id, index, counts) for index, (full_id, counts) in enumerate(period_counts.items())]
    pooled = [period_counts[full_id] for full_id in real_time if full_id in period_counts]
    if pooled:
        frames.append(scope_stats(REAL_TIME_SCOPE, len(period_counts), pd.concat(pooled, ignore_index=True)))
    return pd.concat(frames, ignore_index=True)
