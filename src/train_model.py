import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .config import DATA_DIR, MODEL_PATH, FEATURES_PATH, METRICS_PATH, MODELS_DIR

TARGET_COL = "churn"

NUM_FEATURES = ["tenure_months", "monthly_charges", "total_charges"]
CAT_FEATURES = ["contract_type", "internet_service", "payment_method"]
BOOL_FEATURES = ["paperless_billing"]

def load_data() -> pd.DataFrame:
    path = DATA_DIR / "Telco-Customer-Churn.csv"
    df = pd.read_csv(path)
    return df

def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_FEATURES),
            ("bool", "passthrough", BOOL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ]
    )

    clf = LogisticRegression(max_iter=200)
    pipe = Pipeline(steps=[("preprocess", pre), ("model", clf)])
    return pipe

def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)) if len(y_test) else None,
        "precision": float(precision_score(y_test, y_pred, zero_division=0)) if len(y_test) else None,
        "recall": float(recall_score(y_test, y_pred, zero_division=0)) if len(y_test) else None,
        "f1": float(f1_score(y_test, y_pred, zero_division=0)) if len(y_test) else None,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    # Save artifacts
    joblib.dump(pipe, MODEL_PATH)

    # Save a simple list of expected columns (for input sanity checks later)
    joblib.dump(list(X.columns), FEATURES_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model to:", MODEL_PATH)
    print("Saved metrics to:", METRICS_PATH)
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
