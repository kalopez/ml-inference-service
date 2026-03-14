# ml-inference-service (Churn Prediction)

![Python](https://img.shields.io/badge/python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Docker](https://img.shields.io/badge/docker-containerized-blue)

ml-inference-service is a production-style machine learning inference service that exposes a trained model through a REST API. The project demonstrates how a machine learning model can be trained, packaged, and served as a scalable backend service using modern engineering practices.

The service trains a customer churn prediction model using scikit-learn and provides a FastAPI-based inference endpoint that returns predictions in real time. The application is containerized with Docker, enabling consistent and portable deployment across development and cloud environments. The project is designed with cloud deployment in mind (AWS) and will be extended with CI/CD automation and infrastructure components.

This project focuses on the engineering side of applied AI, emphasizing:

- Reproducible machine learning workflows
- Model artifact management
- API-based inference services
- Containerized deployment using Docker
- Cloud-ready architecture

It is intended as an example of how machine learning models can be integrated into production systems and deployed as reliable backend services.

## Key Features

- Machine learning model training pipeline (scikit-learn)
- REST API for predictions using FastAPI
- Input validation using Pydantic schemas
- Containerized service with Docker
- Reproducible model artifacts
- Clear project structure for ML systems development

---

## Architecture Overview

Client Request
      │
      ▼
FastAPI REST API (/predict)
      │
      ▼
Trained ML Model (scikit-learn)
      │
      ▼
Prediction Response (JSON)

---

## Technologies Used

- Python
- FastAPI
- scikit-learn
- Pandas / NumPy
- Docker
- Uvicorn

---

## Project Goals

This repository demonstrates how to:

- Train and serialize a machine learning model
- Build a REST API for real-time model inference<br>

Future improvements will include:
- Containerize the application for consistent deployment
- Structure an ML system in a maintainable and production-oriented way
- CI/CD pipeline for automated builds and testing
- AWS deployment (EC2 or container services)
- Enhanced model evaluation and monitoring

---

Artifacts saved to:

- `models/churn_model.joblib`
- `models/metrics.json`

## Setup (Local)

### 1) Create virtual environment & install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Train the model

```bash
python -m src.train_model
ls -la models/
cat models/metrics.json
```

### 3) Run the API

**Option A – same terminal (run server in background)**  
Start the server in the background so you can keep using the terminal for curl:

```bash
uvicorn src.app:app --reload --port 8000 &
```

Then run curl in the same terminal. To stop the server later: `kill %1` (or `fg` then Ctrl+C). <br>

**Option B – two terminals**  
In one terminal run `uvicorn src.app:app --reload --port 8000` and leave it running; use the other for curl.

Then open http://127.0.0.1:8000/docs for the interactive API docs. Use port **8000** (or any port ≥ 1024); ports below 1024 require root and will give "Permission denied".

**Test with curl** (server must be running):

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"tenure_months":12,"monthly_charges":70.5,"total_charges":846,"contract_type":"month-to-month","internet_service":"fiber","paperless_billing":true,"payment_method":"electronic-check"}'
```

## Troubleshooting

  **Model artifact not found**  
     * Run: `python -m src.train_model`

  **curl: Failed to connect / Connection refused**  
     * Start the API first in another terminal: `uvicorn src.app:app --reload --port 8000`, then run curl.

  **Port already in use**
     * Run API on 8001: uvicorn src.app:app --reload --port 8001 

## Roadmap

Week 1: Docker
Week 2: AWS- deployment readiness
Week 3: CI/CD automation