# Build context must be the repository root (see docker-compose.yml).
FROM python:3.11-slim

WORKDIR /app

COPY code/deployment/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY code/deployment/api/main.py .
# Bake the trained model into the image so Airflow DooD deploys work
# even when bind-mount host paths would resolve incorrectly.
COPY models/titanic_model.joblib /app/models/titanic_model.joblib

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
