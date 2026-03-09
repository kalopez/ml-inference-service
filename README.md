# ml-inference-service (Churn Prediction)

Production-style ML inference service demonstrating:
- Model training (scikit-learn)
- Model serving (FastAPI)
- Containerization (Docker)
- AWS deployment readiness (Week 2)
- CI automation (Week 3)

## Architecture (Week 1)

Client -> FastAPI (/predict) -> scikit-learn model -> JSON response

Artifacts saved to:
- `models/churn_model.joblib`
- `models/metrics.json`

## Repo Structure
ml-inference-service/
├── data/
├── models/
├── src/
├── tests/
├── pyproject.toml
├── requirements.txt
├── Dockerfile
└── README.md


## Setup (Local)

### 1) Create virtual environment & install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

