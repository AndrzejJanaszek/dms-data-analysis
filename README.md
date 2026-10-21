# dms-data-analysis

Offline analysis of data collected on the concrete plant (weight readings, dispense and refill signals, temperatures).

- `src/dms_analysis/` — data preparation pipeline and analysis code
- `notebooks/` — notebooks for looking at the data (`raw_overview.ipynb`: weight, pins and temperatures of one session on a shared time axis; `signals.ipynb`: signal events after merging pulses, rejected short events, signals before and after cleaning; `event_browser.ipynb`: step through events with a duration, or pairs of events with a gap, in a chosen range; `compare_variants.ipynb`: step through the differences between two rule variants)
- `grafana/` — local Grafana (Docker) with a dashboard over the raw SQLite databases
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

Open http://localhost:3000 (no login). The dashboard `Przegląd surowych danych` reads the databases in `data/` directly through the `frser-sqlite-datasource` plugin. Each database is a data source in `grafana/provisioning/datasources/sqlite.yaml`; a new dataset needs a new entry there and `docker compose restart`. The dashboard lives in `grafana/dashboards/raw-overview.json` and is reloaded from the file, so lasting changes go into that file. Stop with `docker compose down`.

The panels with prepared signals read `data/prepared/signals.db` (not in git), produced by the pipeline for every database listed in `sqlite.yaml` and every rule variant (column `variant`; table `variants` holds the rules, the git commit and the export time). The dashboard variable `Wariant` selects the variant; the histograms of event durations and of gaps between events use it too. Regenerate the file after changing the pipeline or adding data, then refresh the dashboard:

```shell
uv run python -m dms_analysis.export
```
