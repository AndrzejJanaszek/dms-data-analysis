# dms-data-analysis

Offline analysis of data collected on the concrete plant (weight readings, dispense and refill signals, temperatures).

- `src/dms_analysis/` — data preparation pipeline and analysis code
- `notebooks/` — notebooks for looking at the data (`raw_overview.ipynb`: weight, pins and temperatures of one session on a shared time axis)
- `grafana/` — local Grafana (Docker) with a dashboard over the raw SQLite databases
- `data/` — raw datasets
- `legacy/` — the first exploratory script (`main.py`, old setup notes and plots), kept for reference only

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
