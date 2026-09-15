"""
Airflow DAG: Titanic MLOps pipeline
Runs every 5 minutes: data engineering -> model engineering -> deployment.
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

# Inside the Airflow container the project is mounted at /opt/project
PROJECT_ROOT = Path("/opt/project")

default_args = {
    "owner": "pmldl",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="titanic_ml_pipeline",
    description="Automated Titanic MLOps pipeline (data -> model -> deploy)",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="*/5 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["pmldl", "mlops", "titanic"],
) as dag:
    data_engineering = BashOperator(
        task_id="data_engineering",
        bash_command=f"cd {PROJECT_ROOT} && python code/datasets/data_engineering.py",
    )

    model_engineering = BashOperator(
        task_id="model_engineering",
        bash_command=f"cd {PROJECT_ROOT} && python code/models/model_engineering.py",
    )

    deployment = BashOperator(
        task_id="deployment",
        bash_command=f"cd {PROJECT_ROOT} && python code/deployment/deploy.py",
    )

    data_engineering >> model_engineering >> deployment
