from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "01_raw"
TEST_DIR = DATA_DIR / "02_test"

CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


RAW_FILENAME = "customer_data.csv"
TEST_FILENAME = "churn_test.csv"
MODEL_FILENAME = "churn_model.joblib"
METRICS_FILENAME = "churn_metrics.json"

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
RANDOM_STATE = 42
TEST_SIZE = 0.2
