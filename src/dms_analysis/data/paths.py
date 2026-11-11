from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
CATALOG_DIR = PROJECT_DIR / "catalog"
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_PERIODS_DIR = DATA_DIR / "raw-periods"
PREPARED_DIR = DATA_DIR / "prepared"
ALL_PERIODS_DB = PREPARED_DIR / "periods.db"
REVIEW_DIR = DATA_DIR / "review"
DISPENSE_MARKS = REVIEW_DIR / "dispense-marks.csv"


def raw_period_path(full_id: str) -> Path:
    return RAW_PERIODS_DIR / f"{full_id}.db.xz"


def prepared_path(full_id: str) -> Path:
    return PREPARED_DIR / f"{full_id}.db"
