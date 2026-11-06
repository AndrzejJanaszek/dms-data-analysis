import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dms_analysis.between_events_fragments import NO_FRAGMENT, prepare_samples
from dms_analysis.prepared import PreparedSession
from dms_analysis.raw import TIMEZONE
from dms_analysis.synthetic import draw_trials, measure, real_dispenses

STEP = "1min"
MIN_FRAGMENT_STEPS = 60
TEMPERATURE_TOLERANCE = pd.Timedelta(seconds=3)
RATE_HALF_WINDOW_STEPS = 30
INCREMENT_STEPS = (1, 10)
PERCENT = 100
MAIN_METHOD = "mediana 3 s"
SPREAD_SMOOTHING = "30s"
CHANGE_MARGIN = pd.Timedelta(seconds=3)
CHANGE_BANDS = (0.0, 0.05, 0.15, np.inf)
UNCORRECTED = "bez poprawki"
IN_SAMPLE = "współczynniki z całej sesji"
OTHER_DAYS = "współczynniki z pozostałych dni"
CORRECTION_MODELS = {"rozrzut": ["spread"], "rozrzut + szybkość zmian": ["spread", "rate"]}
REAL_DAY_HOURS = tuple(range(9, 16))
REAL_NIGHT_HOURS = (*range(18, 24), *range(0, 8))
DAY = "dzień 9–16"
NIGHT = "noc 18–8"
OTHER = "pozostałe"
ALL = "wszystkie"
FIGURE_HEIGHT = 950
FEATURES = {
    "rozrzut": ["spread"],
    "rozrzut procentowo od poziomu fragmentu": ["spread_load"],
    "rozrzut + średnia": ["spread", "mean"],
    "rozrzut + szybkość zmian średniej": ["spread", "rate"],
    "szybkość zmian średniej": ["rate"],
    "średnia": ["mean"],
}


def temperature_features(session: PreparedSession) -> pd.DataFrame:
    readings = session.data.temperatures
    sensors = readings.pivot_table(index="timestamp", columns="sensor_address", values="value")
    features = pd.DataFrame({"spread": sensors.std(axis=1, ddof=0), "mean": sensors.mean(axis=1)})
    return features.sort_index().reset_index()


def minute_table(session: PreparedSession) -> pd.DataFrame:
    samples = prepare_samples(session).sort_values("timestamp")
    joined = pd.merge_asof(
        samples, temperature_features(session), on="timestamp", direction="nearest", tolerance=TEMPERATURE_TOLERANCE
    )
    calm = joined[(joined["fragment"] != NO_FRAGMENT) & ~joined["spike"]].set_index("timestamp")

    table = (
        calm.groupby("fragment")
        .resample(STEP)
        .agg({"value": "median", "spread": "mean", "mean": "mean"})
        .dropna()
        .reset_index()
    )
    steps = table.groupby("fragment")["value"].transform("size")
    table = table[steps >= MIN_FRAGMENT_STEPS].copy()

    by_fragment = table.groupby("fragment")["mean"]
    later = by_fragment.shift(-RATE_HALF_WINDOW_STEPS)
    earlier = by_fragment.shift(RATE_HALF_WINDOW_STEPS)
    table["rate"] = (later - earlier) * 60 / (2 * RATE_HALF_WINDOW_STEPS)
    table["level"] = table.groupby("fragment")["value"].transform("mean")
    table["spread_load"] = table["spread"] * table["level"] / PERCENT
    table["timestamp"] = table["timestamp"].dt.tz_convert(TIMEZONE)
    return table.reset_index(drop=True)


def fit(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = table.dropna(subset=columns)
    within = _within_fragment(rows, ["value", *columns])
    coefficients, *_ = np.linalg.lstsq(within[columns].to_numpy(), within["value"].to_numpy(), rcond=None)

    fitted = rows["value"] - within["value"] + within[columns].to_numpy() @ coefficients
    result = rows.assign(fitted=fitted, residual=rows["value"] - fitted)
    return _with_coefficients(result, columns, coefficients)


def comparison(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, columns in FEATURES.items():
        fitted = fit(table, columns)
        spread_kg = _within_fragment(fitted, ["value"])["value"]
        rows.append(
            {
                "model": name,
                "współczynniki": fitted.attrs["coefficients"],
                "odch. std wskazań [kg]": round(spread_kg.std()),
                "odch. std reszt [kg]": round(fitted["residual"].std()),
                "R²": round(1 - fitted["residual"].var() / spread_kg.var(), 2),
            }
        )
    return pd.DataFrame(rows)


def fragment_summary(table: pd.DataFrame) -> pd.DataFrame:
    groups = table.groupby("fragment")
    return pd.DataFrame(
        {
            "początek": groups["timestamp"].first(),
            "koniec": groups["timestamp"].last(),
            "minuty": groups.size(),
            "rozrzut maks. [°C]": groups["spread"].max().round(2),
            "zakres wskazań [kg]": groups["value"].agg(lambda values: values.max() - values.min()),
            "r (wskazanie, rozrzut)": groups[["value", "spread"]].apply(lambda x: x["value"].corr(x["spread"])).round(2),
            "r (wskazanie, średnia)": groups[["value", "mean"]].apply(lambda x: x["value"].corr(x["mean"])).round(2),
        }
    ).reset_index()


def figure(fitted: pd.DataFrame, title: str) -> go.Figure:
    result = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            "wskazania (mediana minutowa) i dopasowanie",
            "reszta = wskazanie − dopasowanie",
            "rozrzut temperatur (odch. std 4 czujników)",
            "średnia temperatura",
        ),
    )

    for _, fragment in fitted.groupby("fragment"):
        time = fragment["timestamp"]
        result.add_trace(_line(time, fragment["value"], "wskazanie", "black"), row=1, col=1)
        result.add_trace(_line(time, fragment["fitted"], "dopasowanie", "crimson"), row=1, col=1)
        result.add_trace(_line(time, fragment["residual"], "reszta", "gray"), row=2, col=1)
        result.add_trace(_line(time, fragment["spread"], "rozrzut", "darkorange"), row=3, col=1)
        result.add_trace(_line(time, fragment["mean"], "średnia", "steelblue"), row=4, col=1)

    result.update_yaxes(title_text="kg", row=1, col=1)
    result.update_yaxes(title_text="kg", row=2, col=1)
    result.update_yaxes(title_text="°C", row=3, col=1)
    result.update_yaxes(title_text="°C", row=4, col=1)
    result.update_layout(title=title, height=FIGURE_HEIGHT, showlegend=False)
    return result


def scatter(fitted: pd.DataFrame) -> go.Figure:
    within = _within_fragment(fitted, ["value", "spread"])
    hours = fitted["timestamp"].dt.hour
    result = go.Figure(
        go.Scatter(
            x=within["spread"],
            y=within["value"],
            mode="markers",
            marker={"size": 3, "color": hours, "colorscale": "Twilight", "colorbar": {"title": "godzina"}},
        )
    )
    return result.update_layout(
        title="wskazanie wobec rozrzutu temperatur (odchylenia od średnich we fragmencie)",
        xaxis_title="rozrzut − średni rozrzut we fragmencie [°C]",
        yaxis_title="wskazanie − średnie wskazanie we fragmencie [kg]",
        height=500,
    )


def _within_fragment(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    means = table.groupby("fragment")[columns].transform("mean")
    return table[columns] - means


def _with_coefficients(table: pd.DataFrame, columns: list[str], coefficients: np.ndarray) -> pd.DataFrame:
    table.attrs["coefficients"] = ", ".join(
        f"{column} {_rounded(value)}" for column, value in zip(columns, coefficients, strict=True)
    )
    return table


def _rounded(value: float) -> str:
    return f"{value:.0f}" if abs(value) >= 10 else f"{value:.2f}"


def _line(time: pd.Series, values: pd.Series, name: str, color: str) -> go.Scatter:
    return go.Scatter(x=time, y=values, name=name, mode="lines", line={"color": color, "width": 1})


def increments(table: pd.DataFrame, steps: int) -> pd.DataFrame:
    by_fragment = table.groupby("fragment")
    elapsed = by_fragment["timestamp"].diff(steps)
    changes = pd.DataFrame(
        {
            "fragment": table["fragment"],
            "timestamp": table["timestamp"],
            "value": by_fragment["value"].diff(steps),
            "spread": by_fragment["spread"].diff(steps),
        }
    )
    consecutive = elapsed == pd.Timedelta(STEP) * steps
    return changes[consecutive].reset_index(drop=True)


def increment_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for steps in INCREMENT_STEPS:
        changes = increments(table, steps)
        slope = _slope_through_origin(changes)
        residual = changes["value"] - slope * changes["spread"]
        rows.append(
            {
                "przyrost co [min]": steps,
                "pary": len(changes),
                "nachylenie [kg/°C]": round(slope),
                "korelacja": round(changes["value"].corr(changes["spread"]), 2),
                "odch. std przyrostów wskazań [kg]": round(changes["value"].std()),
                "odch. std reszt [kg]": round(residual.std()),
            }
        )
    return pd.DataFrame(rows)


def increment_scatter(table: pd.DataFrame, steps: int) -> go.Figure:
    changes = increments(table, steps)
    slope = _slope_through_origin(changes)
    edges = np.array([changes["spread"].min(), changes["spread"].max()])

    result = go.Figure(
        [
            go.Scatter(
                x=changes["spread"],
                y=changes["value"],
                mode="markers",
                name="pary minut",
                marker={"size": 3, "color": changes["timestamp"].dt.hour, "colorscale": "Twilight"},
            ),
            go.Scatter(x=edges, y=slope * edges, mode="lines", name=f"{slope:.0f} kg/°C", line={"color": "crimson"}),
        ]
    )
    return result.update_layout(
        title=f"przyrosty co {steps} min: zmiana wskazania wobec zmiany rozrzutu",
        xaxis_title="zmiana rozrzutu [°C]",
        yaxis_title="zmiana wskazania [kg]",
        height=500,
    )


def _slope_through_origin(changes: pd.DataFrame) -> float:
    return float((changes["spread"] * changes["value"]).sum() / (changes["spread"] ** 2).sum())


def coefficients(table: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    rows = table.dropna(subset=columns)
    within = _within_fragment(rows, ["value", *columns])
    values, *_ = np.linalg.lstsq(within[columns].to_numpy(), within["value"].to_numpy(), rcond=None)
    return dict(zip(columns, values.tolist(), strict=True))


def day_coefficients(table: pd.DataFrame) -> pd.DataFrame:
    days = table["timestamp"].dt.date
    rows = []
    for day in sorted(days.unique()):
        row: dict[str, object] = {"dzień": day, "minut w dniu": int((days == day).sum())}
        for model, columns in CORRECTION_MODELS.items():
            for column, value in coefficients(table[days != day], columns).items():
                row[f"{model}: {column} bez tego dnia"] = round(value)
        rows.append(row)
    return pd.DataFrame(rows)


def second_features(session: PreparedSession) -> pd.DataFrame:
    features = temperature_features(session).set_index("timestamp")
    spread = features["spread"].rolling(SPREAD_SMOOTHING).mean()

    mean = features["mean"].resample(STEP).mean()
    half_window = RATE_HALF_WINDOW_STEPS * pd.Timedelta(STEP)
    later = mean.shift(-RATE_HALF_WINDOW_STEPS)
    earlier = mean.shift(RATE_HALF_WINDOW_STEPS)
    rate = (later - earlier) / (2 * half_window / pd.Timedelta(hours=1))
    rate.index = rate.index + pd.Timedelta(STEP) / 2
    both = rate.index.union(features.index)
    second_rate = rate.reindex(both).interpolate(method="time").reindex(features.index)

    return pd.DataFrame({"spread": spread, "rate": second_rate}).reset_index()


def corrected(samples: pd.DataFrame, features: pd.DataFrame, slopes: pd.DataFrame) -> pd.DataFrame:
    joined = pd.merge_asof(samples, features, on="timestamp", direction="nearest", tolerance=TEMPERATURE_TOLERANCE)
    correction = np.zeros(len(samples))
    for column in slopes.columns:
        correction += slopes[column].to_numpy() * joined[column].ffill().bfill().to_numpy()
    return samples.assign(value=samples["value"].to_numpy() - correction)


def variant_name(model: str, held_out: bool) -> str:
    return f"{model}, {OTHER_DAYS if held_out else IN_SAMPLE}"


def variant_order() -> list[str]:
    return [UNCORRECTED, *(variant_name(model, held) for model in CORRECTION_MODELS for held in (False, True))]


def semi_synthetic(session: PreparedSession, table: pd.DataFrame) -> pd.DataFrame:
    samples = prepare_samples(session).sort_values("timestamp").reset_index(drop=True)
    features = second_features(session)
    trials = draw_trials(samples, real_dispenses(session))
    trials["part_of_day"] = trials["start"].dt.hour.map(real_part_of_day)
    trials["spread_change"] = _spread_change(trials, temperature_features(session))

    days = samples["timestamp"].dt.date
    variants = {UNCORRECTED: samples}
    for model, columns in CORRECTION_MODELS.items():
        whole = coefficients(table, columns)
        held_out = {day: coefficients(table[table["timestamp"].dt.date != day], columns) for day in days.unique()}
        variants[variant_name(model, False)] = corrected(
            samples, features, pd.DataFrame([whole] * len(samples), index=samples.index)
        )
        variants[variant_name(model, True)] = corrected(
            samples, features, pd.DataFrame(days.map(held_out).tolist(), index=samples.index)
        )

    results = pd.concat(
        [measure(variant, trials).assign(variant=name) for name, variant in variants.items()], ignore_index=True
    )
    return results.assign(spread_change=results["trial"].map(trials["spread_change"]))


def change_summary(results: pd.DataFrame, method: str = MAIN_METHOD) -> pd.DataFrame:
    chosen = results[results["method"] == method]
    bands = pd.cut(chosen["spread_change"], CHANGE_BANDS, right=False)
    groups = chosen.groupby([bands, "variant"], observed=True, sort=False)["error_kg"]
    summary = pd.DataFrame({"prób": groups.size(), "odchylenie std [kg]": groups.std().round(1)})
    summary.index = summary.index.set_names(["zmiana rozrzutu w odsypie [°C]", "wariant"])
    return summary.unstack("wariant").reindex(columns=variant_order(), level=1).sort_index()


def _spread_change(trials: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    series = features.set_index("timestamp")["spread"]
    before = series.asof(pd.DatetimeIndex(trials["start"] - CHANGE_MARGIN))
    after = series.asof(pd.DatetimeIndex(trials["end"] + CHANGE_MARGIN))
    return pd.Series(np.abs(after.to_numpy() - before.to_numpy()), index=trials.index)


def correction_summary(results: pd.DataFrame, method: str = MAIN_METHOD) -> pd.DataFrame:
    chosen = results[results["method"] == method]
    together = pd.concat([chosen, chosen.assign(part_of_day=ALL)])
    groups = together.groupby(["part_of_day", "variant"], sort=False)["error_kg"]
    summary = pd.DataFrame(
        {
            "prób": groups.size(),
            "błąd średni [kg]": groups.mean().round(1),
            "odchylenie std [kg]": groups.std().round(1),
            "|błąd| mediana [kg]": groups.apply(lambda errors: errors.abs().median()).round(1),
            "|błąd| p95 [kg]": groups.apply(lambda errors: errors.abs().quantile(0.95)).round(1),
        }
    )
    order = pd.MultiIndex.from_product([[DAY, OTHER, NIGHT, ALL], variant_order()])
    return summary.reindex(order).dropna(how="all")


def method_summary(results: pd.DataFrame) -> pd.DataFrame:
    table = results.groupby(["method", "variant"], sort=False)["error_kg"].std().round(1).unstack("variant")
    return table[variant_order()]


def correction_figure(results: pd.DataFrame, method: str = MAIN_METHOD) -> go.Figure:
    chosen = results[results["method"] == method]
    result = go.Figure()
    for name in variant_order():
        errors = chosen[chosen["variant"] == name]
        result.add_trace(go.Box(x=errors["part_of_day"], y=errors["error_kg"], name=name, boxpoints=False))
    return result.update_layout(
        title=f"błąd masy sztucznego odsypu ({method}) bez poprawki i z poprawkami temperaturowymi",
        yaxis_title="błąd [kg]",
        boxmode="group",
        height=500,
    )


def real_part_of_day(hour: int) -> str:
    if hour in REAL_DAY_HOURS:
        return DAY
    if hour in REAL_NIGHT_HOURS:
        return NIGHT
    return OTHER
