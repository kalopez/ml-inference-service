# ml-inference-service (Churn Prediction)

Python
FastAPI
Docker

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

Future improvements will include:

- Build a REST API for real-time model inference
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

## Troubleshooting

**- Model artifact not found**
      * Run: python -m src.train_model

## Roadmap

Week 1: FastAPI service + Docker

Week 2: AWS- deployment readiness
Week 3: CI/CD automation