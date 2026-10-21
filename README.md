# dms-data-analysis

Offline analysis of data collected on the concrete plant (weight readings, dispense and refill signals, temperatures).

- `src/dms_analysis/` — data preparation pipeline and analysis code
- `notebooks/` — notebooks for looking at the data (`raw_overview.ipynb`: weight, pins and temperatures of one session on a shared time axis; `signals.ipynb`: signal events after merging pulses, rejected short events, signals before and after cleaning; `event_browser.ipynb`: step through events with a duration, or pairs of events with a gap, in a chosen range; `compare_variants.ipynb`: step through the differences between two rule variants; `prepared_browser.ipynb`: read-only browser of `prepared.db` — pick the database, session and variant as in Grafana, then step through the events with weight readings and signals)
- `grafana/` — local Grafana (Docker) with a dashboard over the raw SQLite databases and a dashboard over the prepared data
- `data/` — raw datasets
- `legacy/` — the first exploratory script (`main.py`, old setup notes and plots), kept for reference only

## Signal preparation rules

The rules for turning raw pin states into events (merge threshold, minimum duration, optional length-aware merging) are named variants in `VARIANTS` in `src/dms_analysis/signals.py`; `CURRENT_VARIANT` is the one used by default. To change the rules, add a new variant instead of editing an existing one, so earlier results stay reproducible and can be compared with `compare_variants.ipynb`.

## Setup

```shell
uv sync
```

## Notebooks

```shell
uv run jupyter lab notebooks/
```

or open a notebook in VS Code with the `.venv` interpreter as the kernel. The plots use `plotly-resampler`, which reloads detail on zoom, so they need a running kernel.

## Grafana

```shell
cd grafana && docker compose up -d
```

Open http://localhost:3000 (no login). Two dashboards, linked to each other with the same database, session and time range:

- `Przegląd surowych danych` (home) reads the databases in `data/` directly through the `frser-sqlite-datasource` plugin, with the prepared signals, the rejected events and the histograms of event durations and gaps next to the raw data;
- `Przygotowane dane` reads only the prepared database: sessions, weight readings, signal events of the selected variant (accepted and rejected), a table of events in the visible range and temperatures.

 Each database is a data source in `grafana/provisioning/datasources/sqlite.yaml`; a new dataset needs a new entry there and `docker compose restart`. The dashboards live in `grafana/dashboards/` and are reloaded from the files, so lasting changes go into those files. Stop with `docker compose down`.

The prepared data is `data/prepared/prepared.db` (not in git), produced by the pipeline for every database listed in `sqlite.yaml`: tables `sessions`, `weight` and `temperatures` (copied unchanged for now, with a `source` column naming the raw database), `signal_changes` and `signal_events` for every rule variant (column `variant`), and `variants` with the rules, the git commit and the export time. The dashboard variable `Wariant` selects the variant. Regenerate the file after changing the pipeline or adding data, then refresh the dashboard:

```shell
uv run python -m dms_analysis.export
```
