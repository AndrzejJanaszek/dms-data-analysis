# Snapshot of the prepared dataset for analysis

The analysis uses only the main site measurement: database `2026-10-18 obiekt` (now `data/raw/rpi4/2026-10-18/measurement-data-saver.db`, catalogued as period `swietochlowice-1/0002-2026-09-28`), session 9. `prepared-obiekt-sesja-9.db.xz` holds that session cut out of `data/legacy/prepared.db` (until 2026-11-11 `data/prepared/prepared.db`), kept in git as a backup and compressed with xz.

## prepared-obiekt-sesja-9.db.xz

- Period: 2026-09-23 13:00 – 2026-10-04 16:34, local time (Europe/Warsaw) from the Raspberry Pi clock, which has no RTC, so the dates may be shifted.
- Contents: the same tables as `prepared.db`, limited to that session: 962 610 weight readings, no temperatures (no sensors in this session), signal changes and 1 278 events across rule variants v1–v4 (current: v4), and the `variants` table.
- Prepared by commit `6a0ca92` (stored in the `variants` table); exported 2026-10-21.
- Size: 8.3 MB compressed, 86.7 MB unpacked; SHA-256 of the unpacked file: `90d9706014a92b7f019b0e9cdf5139bad7696b4c29e2f43c587f53010b119122`.

Recreate after changing the pipeline (`ANALYSIS_SOURCE` and `ANALYSIS_SESSION` in `dms_analysis.snapshot` select the session):

```shell
uv run python -m dms_analysis.export
uv run python -m dms_analysis.snapshot
```

Unpack (keeps the archive); the file can be opened in `notebooks/prepared_browser.ipynb` by passing its path to `PreparedBrowser`:

```shell
xz -dkc data/legacy/snapshots/prepared-obiekt-sesja-9.db.xz > data/legacy/obiekt-sesja-9.db
```
