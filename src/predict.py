import joblib
import pandas as pd

from .config import MODEL_PATH
from .schema import PredictRequest

def predict_one(req: PredictRequest) -> tuple[float, int]:
    model = joblib.load(MODEL_PATH)

    row = {
        "tenure_months": req.tenure_months,
        "monthly_charges": req.monthly_charges,
        "total_charges": req.total_charges,
        "contract_type": req.contract_type,
        "internet_service": req.internet_service,
        "paperless_billing": req.paperless_billing,
        "payment_method": req.payment_method,
    }

    X = pd.DataFrame([row])

    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(proba >= 0.5)
    return proba, pred
