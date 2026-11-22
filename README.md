# dms-data-analysis

Offline analysis of data collected on the concrete plant (weight readings, dispense and refill signals, temperatures).

- `src/dms_analysis/` — data preparation pipeline and analysis code
- `notebooks/` — notebooks for looking at the data (`raw_overview.ipynb`: weight, pins and temperatures of one session on a shared time axis; `signals.ipynb`: signal events after merging pulses, rejected short events, signals before and after cleaning; `event_browser.ipynb`: step through events with a duration, or pairs of events with a gap, in a chosen range; `compare_variants.ipynb`: step through the differences between two rule variants; `prepared_browser.ipynb`: read-only browser of `prepared.db` — pick the database, session and variant as in Grafana, then step through the events with weight readings and signals)
- `grafana/` — local Grafana (Docker) with a dashboard over the raw SQLite databases and a dashboard over the prepared data
- `catalog/` — data catalog (TOML, the only hand-edited data files): raw dumps and their checksums, sites, hardware configurations and recording periods; see [Data catalog](#data-catalog)
- `src/dms_analysis/data/` — data preparation stage (catalog, dumps, period slices, prepared period databases, check report helpers); `notebooks/data/period_check.ipynb` — check report of one prepared period; `notebooks/data/dispense_review.ipynb` — step through the dispenses of a period (◀ / ▶; all, suspicious only, or marked bad), suspicious ones first (rejected or without a weight drop), mark each as good or bad with a link to Grafana; the verdicts are saved to `data/review/dispense-marks.csv` (review record; a bad accepted dispense goes to the catalog as a `dispense` issue)
- `data/raw/` — raw dumps (not in git, kept on Google Drive `INZYNIERKA/dms-data/raw/`); `data/raw-periods/` — raw slice of each period, xz-compressed, in git; `data/prepared/` — generated, not in git
- `data/legacy/` — files of the old pipeline: the July CSVs (`df_1.csv`, `df_2.csv`, `days/`), `prepared.db` and `obiekt-sesja-9.db` (not in git), `snapshots/` (compressed copy of the old prepared dataset of session 9, made by `uv run python -m dms_analysis.snapshot`; details in its README)
- `tests/` — pytest tests of the data preparation stage (`uv run pytest`)
- `legacy/` — the first exploratory script (`main.py`, old setup notes and plots), kept for reference only

## Signal preparation rules

The prepared periods (`dms_analysis.data`) use only `CURRENT_VARIANT`. The rules for turning raw pin states into events (merge threshold, minimum duration, optional length-aware merging) are named variants in `VARIANTS` in `src/dms_analysis/signals.py`; `CURRENT_VARIANT` is the one used by default. To change the rules, add a new variant instead of editing an existing one, so earlier results stay reproducible and can be compared with `compare_variants.ipynb`.

## Data catalog

The analysis works on **periods**: continuous recordings from one site, described in the catalog and prepared one file per period. Period id: `<site>/<NNNN>-<start date>` (e.g. `swietochlowice-1/0002-2026-09-28`, `NNNN-unknown` when the date is not known); the number is sequential per site and the id never changes.

- `catalog/raw.toml` — `[[dump]]` per raw dump: device, pull date, file under `data/raw/`, SHA-256, program, note, `contained_in` (a newer dump that holds all its rows).
- `catalog/<site>.toml` — `[site]`, versioned `[[hardware]]` (silo, meter, division, `pins = { 5 = "dispense", 4 = "refill" }`, `temperature_sensors` with number, address and position) and `[[period]]`: source (dump + session, or dump + `first_id`/`last_id` for the old program without sessions), time status `real` | `estimated` | `unknown` with `clock_offset_h` (real = clock + offset; `unknown` keeps the device clock, offset 0) and `time_uncertainty_h`, `start`/`end` after correction (local time with offset), `checked`, `[period.channels]` (`weight`, `dispense`, `refill`, `temperatures`: `ok` | `flickering` | `broken` | `not_connected` | `no_values`), `[[period.issue]]` (problems in part of a period) and `[[period.manual_event]]` (events entered by hand, `end` omitted = open until the end of the period).

The catalog is validated when loaded (clear error in Polish). Scripts never rewrite it; they print blocks to paste.

Commands (`uv run python -m dms_analysis.data <command>`):

| Command | What it does |
|---|---|
| `add-dump <file> --device rpi4 --pulled 2026-11-09` | copies a dump to `data/raw/<device>/<date>/` (a non-empty `-wal` next to it is merged into the copy), prints the `[[dump]]` block with SHA-256 |
| `verify` | checks the checksums of all dumps present locally, lists missing ones and any `-wal`/`-shm` next to a dump |
| `sessions <dump> [--site S --session N] [--all]` | sessions of a dump not yet in the catalog (segments by `id` for the old program) and a draft `[[period]]` block |
| `extract [--period ID]` | creates missing raw slices `data/raw-periods/<site>/<id>.db.xz` (never overwrites) |
| `prepare [--period ID]` | builds `data/prepared/<site>/<id>.db` from the slice and the catalog, then `data/prepared/periods.db` (all periods, for Grafana) |

`prepare` reads only the slices, so a fresh clone needs just `uv sync` and `prepare`. Dumps and slices are opened read-only with `immutable=1`.

Prepared period database (times in UTC epoch seconds after the clock correction): `period` (catalog fields, channel states, variant, commit, SHA-256 of the catalog block with its hardware), `weight`, `temperatures` (sensor number from the hardware; DS18B20 error codes, malformed addresses and sensors outside the hardware removed), `removed` (what was removed and why), `signal_changes` and `signal_events` (current rule variant; `broken` channel = manual events only, `not_connected` = nothing; manual events have `synthetic = 1`), `issues`, `parts` (split at a weight gap longer than 60 s). The check report and later analyses warn when the catalog changed since the last `prepare`.

New data:

1. `add-dump`, paste the block into `catalog/raw.toml`, upload the dump to Drive.
2. `sessions <dump> --site … --session …`, paste the draft into the site file, set time status, offset, channel states, issues and manual events.
3. `extract`, `prepare`.
4. Run `notebooks/data/period_check.ipynb` for the period (variable `PERIOD`, or `PERIOD=… uv run jupyter nbconvert --to notebook --execute …`) and look at it in Grafana (`Okresy`); then set `checked = true` and run `prepare` again.

## Analysis stage 2: event values

`uv run python -m dms_analysis.events compute [--period ID]` (after `prepare`) computes, for every checked period with a dispense signal, `data/results/<site>/<id>/event_values.db` (outside git):

- `dispenses` and `dispense_values`: mass of every accepted dispense in 9 variants (nearest sample, median, mean of a 3 s window × gap after the signal end d = 0, 2, 3 s; mass = before − after), `excluded` = `okno` (the widest window touches another accepted event, a catalog issue, a part boundary or misses samples) or `kolizja` (overlaps a refill), `no_drop` per variant, gaps to the neighbouring events. Rejected dispenses are signal errors and are not neighbours.
- `refills`: refill mass from single samples and from 3 s medians, dispenses during the refill.
- `run`: commit, catalog hash and `prepared_at` of the prepared database, parameters.

Results are reviewed in `notebooks/events/event_values.ipynb` (summary, single-dispense browser, mass and variant plots, refills, first and last dispense hours for the day/night boundaries). The semi-synthetic ramp test was removed in stage 3 (replaced by the grid test). Code: `dms_analysis.events`, plot style `dms_analysis.style`. Decisions: workspace `praca-inzynierska/zakres.md` §7, plan `plans/dms-data-analysis/etap2-wartosci-zdarzen-plan.md`.

## Analysis stage 3: error measures

`uv run python -m dms_analysis.errors <command>` (after `prepare`; code in `dms_analysis.errors`, results outside git). Real events = accepted by the rules plus dispenses rejected by the rules but marked `dobry` in `data/review/dispense-marks.csv`, minus catalog `false_event` issues. Day = 5:00–22:00 local time, night otherwise.

- `grid [--period ID]` (~4 min): fake dispense without mass at every whole second of the event-free fragments (60 s clear of real events, no catalog issues, no sample gaps > 1.5 s), durations T = 10, 20, 30, 45, 60, 90 s, 9 variants; error = before − after. Per period `data/results/<site>/<id>/errors.db` (`run`, `days`, `grid_counts` = error counts per date, hour, T, d, method, `grid_skipped` = starts with a window of fewer than 3 samples, `grid_worst` = 20 worst starts per T and day part, at least 120 s apart); per site `data/results/<site>/errors_summary.db` (`grid_stats`: measures with 95 % percentile intervals from a bootstrap over days, B = 2000, per period and for all real-time periods together). Notebook `notebooks/errors/single_dispense.ipynb` (also relative error = error / typical dispense mass and error versus T on log axes, module `errors.scaling`).
- `balance [--period ID]`: daily mass balance R = [W(end) − W(start)] − Σ refills + Σ dispenses, night to night (median 0:00–4:00) and 5–22 (medians 4:50–5:00 and 22:00–22:10); refills as blocks with the dispenses inside; groups control / without refill / with refill. Per period `data/results/<site>/<id>/balance.db`. Notebook `notebooks/errors/mass_balance.ipynb`.
- `schedule`: every real working day (full day of real-time periods; two periods are joined when the gap is < 300 s and the weight change < 100 kg) replayed on every event-free day at the same clock times; E = sum of the dispense errors, Y = sum of the real dispense masses, two-way bootstrap over schedules and base days. Per site `data/results/<site>/schedule.db`. Notebook `notebooks/errors/day_sum.ipynb`.

Plan: workspace `plans/dms-data-analysis/etap3-miary-bledu-plan.md`.

## Setup

```shell
uv sync
```

## Notebooks

```shell
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 -p CPUWeight=50 uv run jupyter lab notebooks/
```

or open a notebook in VS Code with the `.venv` interpreter as the kernel. The plots use `plotly-resampler`, which reloads detail on zoom, so they need a running kernel.

The memory limit makes the kernel OOM kill Jupyter instead of freezing the whole machine (the VM has no swap). VS Code and Firefox get their limits from user systemd drop-ins (`scripts/vm-limits/` in the workspace).

## Grafana

```shell
cd grafana && docker compose up -d
```

Open http://localhost:3000 (no login). The `Okresy` dashboard reads `data/prepared/periods.db` (data source `okresy`): table of periods (click: select the period and its time range), weight readings, raw and cleaned signals, temperatures, events in the visible range; catalog issues, manual events and local midnight (Europe/Warsaw, so 23:00 or 22:00 UTC) as annotations; a second tab `Wskazania i temperatury` shows only the weight readings with the temperatures right below them. The links `Czas lokalny` and `UTC` at the top switch the time axis between Europe/Warsaw and UTC (same period and range). The period label includes the time status. The older dashboards, linked to each other with the same database, session and time range:

- `Przegląd surowych danych` (home) reads the databases in `data/` directly through the `frser-sqlite-datasource` plugin, with the prepared signals, the rejected events and the histograms of event durations and gaps next to the raw data;
- `Przygotowane dane` reads only the prepared database: sessions, weight readings, signal events of the selected variant (accepted and rejected), a table of events in the visible range and temperatures;
- `Wskazania, temperatury i sygnały` reads a raw database directly: weight readings, then temperatures (one line per sensor, DS18B20 error codes 85 and −127 and malformed sensor addresses left out), then raw pins, on one time axis.

 Each database is a data source in `grafana/provisioning/datasources/sqlite.yaml`; raw dumps are opened with `pathOptions: mode=ro&immutable=1` (without it the plugin creates `-shm`/`-wal` next to them; with a read-only file it falls back to a mode that also creates them, so dumps stay writable and `verify` checks the sums; a path containing `grafana.db` is blocked by the plugin); a new dataset needs a new entry there and `docker compose restart`. The dashboards live in `grafana/dashboards/` and are reloaded from the files, so lasting changes go into those files. `periods-overview.json` is in the dashboard schema v2 (tabs need Grafana 13); edit it in the UI and export the v2 JSON, or edit the file. Stop with `docker compose down`.

The old prepared data is `data/legacy/prepared.db` (not in git), produced by the pipeline for every database listed in `sqlite.yaml`: tables `sessions`, `weight` and `temperatures` (with a `source` column naming the raw database; temperatures without the DS18B20 error codes 85 and −127 and malformed addresses, with the sensor number from `src/dms_analysis/sensors.py`), `signal_changes` and `signal_events` for every rule variant (column `variant`), `variants` with the rules, the git commit and the export time, and `sensors` (address, number, sides of the silo). `FIRST_NEW_SESSION` in `export.py` skips sessions of a newer database that are already exported from an older one. A pin whose signal is broken in a session can be replaced by events entered by hand (`MANUAL_EVENTS` in `src/dms_analysis/manual_events.py`, times in UTC); they apply to every variant and are marked with `synthetic = 1`, so they show where the event was, not when the real signal switched. The dashboard variable `Wariant` selects the variant. Regenerate the file after changing the pipeline or adding data, then refresh the dashboard:

```shell
uv run python -m dms_analysis.export
```
