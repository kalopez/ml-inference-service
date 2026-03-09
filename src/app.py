from fastapi import FastAPI, HTTPException
from .schema import PredictRequest, PredictResponse
from .predict import predict_one
from .config import MODEL_PATH

app = FastAPI(title="ML Inference Service", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=500, detail="Model artifact not found. Run training first.")
    proba, pred = predict_one(req)
    return PredictResponse(churn_probability=proba, churn_prediction=pred)
