# PMLDL Assignment 1: Deployment (Titanic MLOps Pipeline)

Automated MLOps pipeline with three stages:

1. **Data engineering** — load / clean / split the Titanic dataset  
2. **Model engineering** — feature engineering, train RandomForest, evaluate, log with MLflow, save model  
3. **Deployment** — FastAPI model API + Streamlit app in **separate Docker containers**

The full pipeline is scheduled to run **every 5 minutes** (Airflow DAG or lightweight local scheduler).

## Dataset

[Titanic](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) passenger survival data (not from the forbidden lab datasets).

Raw file: `data/raw/titanic.csv`

## Repository structure

```text
├── code
│   ├── datasets/          # Stage 1
│   ├── models/            # Stage 2
│   └── deployment/
│       ├── api/           # FastAPI + Dockerfile
│       ├── app/           # Streamlit + Dockerfile
│       ├── deploy.py
│       └── docker-compose.yml
├── data
│   ├── raw/
│   └── processed/
├── models/                # packaged model (.joblib), created by pipeline
├── notebooks/
├── services/airflow/
│   ├── dags/              # Airflow DAG (every 5 minutes)
│   └── docker-compose.yml
├── run_pipeline.py        # run all stages once
├── schedule_pipeline.py   # local 5-minute scheduler
└── requirements.txt
```

## Prerequisites

- Python 3.11+
- Docker Desktop (required for API + app)
- Git

## Setup

```bash
git clone https://github.com/Telman3000/pmldl-assignment-1-deployment.git
cd pmldl-assignment-1-deployment

python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Run the pipeline once

This runs data → model → Docker deployment:

```bash
python run_pipeline.py
```

After a successful run:

| Service | URL |
|---------|-----|
| FastAPI docs | http://localhost:8000/docs |
| Streamlit app | http://localhost:8501 |
| Health check | http://localhost:8000/health |

Stop containers:

```bash
docker compose -f code/deployment/docker-compose.yml down
```

## Automation every 5 minutes

### Option A — Airflow (recommended for the assignment)

```bash
cd services/airflow
docker compose up --build -d
```

- Airflow UI: http://localhost:8080  
- Login: `admin` / `admin`  
- DAG: `titanic_ml_pipeline` (schedule `*/5 * * * *`)

The DAG tasks call the same Python stages and rebuild/restart the API + app containers.

Stop Airflow:

```bash
docker compose down
```

### Option B — Lightweight local scheduler

```bash
python schedule_pipeline.py
```

Runs the full pipeline immediately, then every 5 minutes. Stop with `Ctrl+C`.

## Run stages separately

```bash
python code/datasets/data_engineering.py
python code/models/model_engineering.py
python code/deployment/deploy.py
```

## Model note

The trained model file (`models/titanic_model.joblib`) is produced by Stage 2 and is listed in `.gitignore` if you prefer not to commit binaries. Generate it by running the pipeline.

MLflow metrics/artifacts are stored under `mlruns/`.

## API example

```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"Pclass\":3,\"Sex\":\"male\",\"Age\":22,\"SibSp\":1,\"Parch\":0,\"Fare\":7.25,\"Embarked\":\"S\"}"
```

## Grading checklist

- [x] Data engineering (clean + train/test split)
- [x] Model engineering (train, metrics, packaged model, MLflow)
- [x] Deployment (API + app in separate Docker containers)
- [x] Automated schedule (~every 5 minutes)
- [x] Logical repository structure + README
