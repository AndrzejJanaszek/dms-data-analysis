import numpy as np
import pandas as pd

BOOTSTRAP_SAMPLES = 2000
CONFIDENCE = 0.95
SEED = 20261117

MEAN = "mean"
STD = "std"
RMSE = "rmse"
MAE = "mae"
P50 = "p50"
P95 = "p95"
P99 = "p99"
MAX = "max"
MEASURES = (MEAN, STD, RMSE, MAE, P50, P95, P99, MAX)
PERCENTILES = {P50: 50.0, P95: 95.0, P99: 99.0}
WITHOUT_INTERVAL = (MAX,)

SUMMARY_COLUMNS = ["measure", "value_kg", "low_kg", "high_kg", "starts", "days", "max_day"]


def measures(values: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    weights = np.atleast_2d(weights).astype(float)
    total = weights.sum(axis=1)
    mean = weights @ values / total
    centered = values[None, :] - mean[:, None]

    result = {
        MEAN: mean,
        STD: np.sqrt((weights * centered**2).sum(axis=1) / (total - 1)),
        RMSE: np.sqrt(weights @ values**2 / total),
        MAE: weights @ np.abs(values) / total,
    }

    sizes = np.abs(values)
    order = np.argsort(sizes, kind="stable")
    ordered_sizes = sizes[order]
    ordered_weights = weights[:, order]
    cumulative = np.cumsum(ordered_weights, axis=1)
    for name, percentile in PERCENTILES.items():
        index = np.argmax(cumulative >= percentile / 100 * total[:, None], axis=1)
        result[name] = ordered_sizes[index]

    last = ordered_weights.shape[1] - 1 - np.argmax(ordered_weights[:, ::-1] > 0, axis=1)
    result[MAX] = ordered_sizes[last]
    return result


def unit_histograms(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unit_codes, units = pd.factorize(frame["unit"], sort=True)
    value_codes, values = pd.factorize(frame["error_kg"], sort=True)
    histograms = np.zeros((len(units), len(values)))
    np.add.at(histograms, (unit_codes, value_codes), frame["starts"].to_numpy(dtype=float))
    return np.asarray(units), np.asarray(values, dtype=float), histograms


def resampled_weights(histograms: np.ndarray, rng: np.random.Generator, samples: int) -> np.ndarray:
    return multiplicities(len(histograms), rng, samples) @ histograms


def summary(frame: pd.DataFrame, rng: np.random.Generator, samples: int = BOOTSTRAP_SAMPLES) -> pd.DataFrame:
    units, values, histograms = unit_histograms(frame)
    point = measures(values, histograms.sum(axis=0))
    spread = measures(values, resampled_weights(histograms, rng, samples))
    tail = (1 - CONFIDENCE) / 2 * 100

    rows = []
    for name in MEASURES:
        low, high = np.nan, np.nan
        if name not in WITHOUT_INTERVAL:
            low, high = np.percentile(spread[name], [tail, 100 - tail])
        rows.append(
            {
                "measure": name,
                "value_kg": float(point[name][0]),
                "low_kg": float(low),
                "high_kg": float(high),
                "starts": int(histograms.sum()),
                "days": len(units),
                "max_day": _max_unit(units, values, histograms) if name == MAX else "",
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _max_unit(units: np.ndarray, values: np.ndarray, histograms: np.ndarray) -> str:
    reached = np.where(histograms > 0, np.abs(values)[None, :], -np.inf).max(axis=1)
    return str(units[np.argmax(reached)])


def point_measures(frame: pd.DataFrame) -> dict[str, float]:
    values = frame["error_kg"].to_numpy(dtype=float)
    result = measures(values, frame["starts"].to_numpy(dtype=float))
    return {name: float(value[0]) for name, value in result.items()}


def multiplicities(count: int, rng: np.random.Generator, samples: int) -> np.ndarray:
    draws = rng.integers(0, count, (samples, count))
    result = np.zeros((samples, count))
    np.add.at(result, (np.repeat(np.arange(samples), count), draws.ravel()), 1.0)
    return result


def two_way_weights(
    rows: np.ndarray, columns: np.ndarray, row_count: int, column_count: int, rng: np.random.Generator, samples: int
) -> np.ndarray:
    row_draws = multiplicities(row_count, rng, samples)
    column_draws = multiplicities(column_count, rng, samples)
    return row_draws[:, rows] * column_draws[:, columns]


def two_way_summary(
    frame: pd.DataFrame, row_count: int, column_count: int, rng: np.random.Generator, samples: int = BOOTSTRAP_SAMPLES
) -> pd.DataFrame:
    keys = frame["row"].to_numpy(dtype=int) * column_count + frame["column"].to_numpy(dtype=int)
    pair_codes, pairs = pd.factorize(keys, sort=True)
    value_codes, values = pd.factorize(frame["value"], sort=True)
    histograms = np.zeros((len(pairs), len(values)))
    np.add.at(histograms, (pair_codes, value_codes), frame["count"].to_numpy(dtype=float))

    rows, columns = np.divmod(np.asarray(pairs, dtype=int), column_count)
    values = np.asarray(values, dtype=float)
    point = measures(values, histograms.sum(axis=0))
    weights = two_way_weights(rows, columns, row_count, column_count, rng, samples) @ histograms
    kept = weights.sum(axis=1) > 1
    spread = measures(values, weights[kept])
    tail = (1 - CONFIDENCE) / 2 * 100

    result = []
    for name in MEASURES:
        low, high = np.nan, np.nan
        if name not in WITHOUT_INTERVAL:
            low, high = np.percentile(spread[name], [tail, 100 - tail])
        result.append(
            {
                "measure": name,
                "value": float(point[name][0]),
                "low": float(low),
                "high": float(high),
                "count": int(histograms.sum()),
                "pairs": len(pairs),
            }
        )
    return pd.DataFrame(result)
