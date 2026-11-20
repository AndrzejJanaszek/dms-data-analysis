from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from dms_analysis.errors import stats
from dms_analysis.errors.fragments import DAY, DAY_PARTS, period_dates
from dms_analysis.errors.grid import ERROR_DECIMALS
from dms_analysis.errors.schedule import Base, LoadedPeriod, Schedule, bases, hours_of, replay, schedules
from dms_analysis.events.values import BASE_DELAY_S, BASE_METHOD, DELAYS_S, METHODS

DAY_SUM = "E [kg]"
DAY_SHARE = "E / Y [%]"
SINGLE = "odtworzony odsyp"
ALL_SCHEDULES = "wszystkie pary"


@dataclass(frozen=True)
class ReplayResult:
    schedules: pd.DataFrame
    bases: pd.DataFrame
    pairs: pd.DataFrame
    pair_errors: pd.DataFrame
    replayed: pd.DataFrame
    stats: pd.DataFrame
    ratio: pd.DataFrame


def site_days(periods: list[LoadedPeriod]) -> list[date]:
    days = sorted({day for period in periods for day in period_dates(period.data)})
    if not days:
        return []
    return [days[0] + timedelta(days=offset) for offset in range((days[-1] - days[0]).days + 1)]


def schedule_table(found: list[Schedule]) -> pd.DataFrame:
    rows = []
    for index, schedule in enumerate(found):
        dispenses = schedule.dispenses
        outside = dispenses[~dispenses["in_refill"]]
        rows.append(
            {
                "schedule": index,
                "date": schedule.day.isoformat(),
                "dispenses": len(dispenses),
                "in_refill": int(dispenses["in_refill"].sum()),
                "y_kg": float(outside["mass_kg"].sum()),
                "periods": ", ".join(schedule.periods),
            }
        )
    return pd.DataFrame(rows)


def base_table(found: list[Base]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"base": index, "date": base.day.isoformat(), "period": base.period.data.full_id}
            for index, base in enumerate(found)
        ]
    )


def compute(periods: list[LoadedPeriod]) -> ReplayResult:
    days = site_days(periods)
    found_schedules = schedules(periods, days)
    found_bases = bases(periods, days)
    schedule_rows = schedule_table(found_schedules)

    pairs, pair_errors, replayed = [], [], []
    for schedule_index, schedule in enumerate(found_schedules):
        y_kg = float(schedule_rows.loc[schedule_index, "y_kg"])
        for base_index, base in enumerate(found_bases):
            result = replay(schedule, base)
            base_error = float(result.errors[(BASE_METHOD, BASE_DELAY_S)].sum()) if result.valid else np.nan
            pairs.append(
                {
                    "schedule": schedule_index,
                    "base": base_index,
                    "valid": int(result.valid),
                    "reason": result.reason,
                    "blocked": result.blocked,
                    "dispenses": len(schedule.dispenses),
                    "e_kg": base_error,
                    "e_share": 100 * base_error / y_kg if y_kg else np.nan,
                }
            )
            if not result.valid:
                continue
            hours = hours_of(result.starts)
            for (method, delay_s), errors in result.errors.items():
                key = {"schedule": schedule_index, "base": base_index, "method": method, "delay_s": delay_s}
                pair_errors.append({**key, "e_kg": float(errors.sum()), "e_share": 100 * float(errors.sum()) / y_kg})
                replayed.append(
                    pd.DataFrame(
                        {
                            **key,
                            "start": result.starts,
                            "hour": hours,
                            "duration_s": result.durations,
                            "error_kg": np.round(errors, ERROR_DECIMALS),
                        }
                    )
                )

    pair_table = pd.DataFrame(pairs)
    error_table = pd.DataFrame(pair_errors)
    replay_table = pd.concat(replayed, ignore_index=True) if replayed else pd.DataFrame()
    shape = (len(found_schedules), len(found_bases))
    return ReplayResult(
        schedules=schedule_rows,
        bases=base_table(found_bases),
        pairs=pair_table,
        pair_errors=error_table,
        replayed=replay_table,
        stats=all_stats(error_table, replay_table, shape),
        ratio=ratio_table(pair_table, schedule_rows, replay_table),
    )


def _seed(target_index: int, method: str, delay_s: float, part_index: int = 0) -> np.random.Generator:
    return np.random.default_rng([stats.SEED, 3, target_index, METHODS.index(method), int(delay_s * 10), part_index])


def all_stats(errors: pd.DataFrame, replayed: pd.DataFrame, shape: tuple[int, int]) -> pd.DataFrame:
    frames = []
    for method in METHODS:
        for delay_s in DELAYS_S:
            chosen = errors[(errors["method"] == method) & (errors["delay_s"] == delay_s)]
            for target_index, (target, column) in enumerate(((DAY_SUM, "e_kg"), (DAY_SHARE, "e_share"))):
                frame = pd.DataFrame(
                    {"row": chosen["schedule"], "column": chosen["base"], "value": chosen[column], "count": 1}
                )
                table = stats.two_way_summary(frame, *shape, _seed(target_index, method, delay_s))
                frames.append(table.assign(target=target, part="", method=method, delay_s=delay_s))

            singles = replayed[(replayed["method"] == method) & (replayed["delay_s"] == delay_s)]
            singles = singles.assign(part=np.where((singles["hour"] >= 5) & (singles["hour"] < 22), DAY, DAY_PARTS[1]))
            for part_index, part in enumerate(DAY_PARTS):
                subset = singles[singles["part"] == part]
                if subset.empty:
                    continue
                counted = subset.groupby(["schedule", "base", "error_kg"]).size().rename("count").reset_index()
                frame = counted.rename(columns={"schedule": "row", "base": "column", "error_kg": "value"})
                table = stats.two_way_summary(frame, *shape, _seed(2, method, delay_s, part_index))
                frames.append(table.assign(target=SINGLE, part=part, method=method, delay_s=delay_s))
    return pd.concat(frames, ignore_index=True)


def ratio_table(pairs: pd.DataFrame, schedule_rows: pd.DataFrame, replayed: pd.DataFrame) -> pd.DataFrame:
    base = replayed[(replayed["method"] == BASE_METHOD) & (replayed["delay_s"] == BASE_DELAY_S)]
    day = base[(base["hour"] >= 5) & (base["hour"] < 22)]
    sigma = float(day["error_kg"].std())
    valid = pairs[pairs["valid"] == 1]

    rows = []
    for schedule, group in valid.groupby("schedule"):
        count = int(group["dispenses"].iloc[0])
        spread = float(group["e_kg"].std())
        rows.append(
            {
                "schedule": schedule_rows.loc[schedule, "date"],
                "dispenses": count,
                "pairs": len(group),
                "std_e_kg": spread,
                "sigma_sqrt_n_kg": sigma * np.sqrt(count),
                "ratio": spread / (sigma * np.sqrt(count)),
            }
        )
    scaled = valid["e_kg"] / np.sqrt(valid["dispenses"])
    rows.append(
        {
            "schedule": ALL_SCHEDULES,
            "dispenses": int(valid["dispenses"].sum()),
            "pairs": len(valid),
            "std_e_kg": float(valid["e_kg"].std()),
            "sigma_sqrt_n_kg": np.nan,
            "ratio": float(scaled.std()) / sigma,
        }
    )
    return pd.DataFrame(rows).assign(sigma_kg=sigma)
