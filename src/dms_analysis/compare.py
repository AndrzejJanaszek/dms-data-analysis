import pandas as pd

UNCHANGED = "bez zmian"
STATUS_CHANGED = "zmiana statusu"
BOUNDS_CHANGED = "zmienione granice"
MERGED = "połączone w B"
SPLIT = "rozdzielone w B"
REGROUPED = "przegrupowane"
CATEGORIES = [STATUS_CHANGED, BOUNDS_CHANGED, MERGED, SPLIT, REGROUPED, UNCHANGED]


def compare_events(events_a: pd.DataFrame, events_b: pd.DataFrame) -> pd.DataFrame:
    tagged = pd.concat([events_a.assign(version="A"), events_b.assign(version="B")], ignore_index=True)
    clusters = []

    for pin, pin_events in tagged.groupby("pin"):
        ordered = pin_events.sort_values(["start", "version"])
        cluster_end = ordered["end"].cummax().shift()
        number = (ordered["start"] >= cluster_end).cumsum()

        for _, members in ordered.groupby(number):
            a = members[members["version"] == "A"]
            b = members[members["version"] == "B"]
            clusters.append(
                {
                    "pin": pin,
                    "role": members["role"].iloc[0],
                    "start": members["start"].min(),
                    "end": members["end"].max(),
                    "category": _category(a, b),
                    "a": a.drop(columns="version"),
                    "b": b.drop(columns="version"),
                }
            )

    return pd.DataFrame(clusters).sort_values("start", ignore_index=True)


def summary(clusters: pd.DataFrame) -> pd.DataFrame:
    table = clusters.groupby(["role", "category"]).size().unstack("category", fill_value=0)
    return table.reindex(columns=[column for column in CATEGORIES if column in table.columns])


def _category(a: pd.DataFrame, b: pd.DataFrame) -> str:
    if len(a) == 1 and len(b) == 1:
        same_bounds = a["start"].iloc[0] == b["start"].iloc[0] and a["end"].iloc[0] == b["end"].iloc[0]
        if not same_bounds:
            return BOUNDS_CHANGED
        return UNCHANGED if a["rejected"].iloc[0] == b["rejected"].iloc[0] else STATUS_CHANGED
    if len(b) == 1:
        return MERGED
    if len(a) == 1:
        return SPLIT
    return REGROUPED
