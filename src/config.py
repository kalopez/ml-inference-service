from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "churn_model.joblib"
FEATURES_PATH = MODELS_DIR / "feature_columns.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
