"""
Stage 2 — Model Engineering
Feature engineering, train a classifier, evaluate, log with MLflow, save model.
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "titanic_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.txt"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]
TARGET_COLUMN = "Survived"


def load_processed_data(
    train_path: Path = TRAIN_PATH, test_path: Path = TEST_PATH
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Processed train/test files not found. Run data engineering first."
        )
    return pd.read_csv(train_path), pd.read_csv(test_path)


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def evaluate(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def run_model_engineering() -> dict:
    train_df, test_df = load_processed_data()

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x_train = train_df[feature_cols]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df[feature_cols]
    y_test = test_df[TARGET_COLUMN]

    pipeline = build_pipeline()

    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("titanic-survival")

    with mlflow.start_run(run_name="random_forest"):
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        metrics = evaluate(y_test, y_pred)

        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": 100,
                "max_depth": 6,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)

        metrics_text = "\n".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        METRICS_PATH.write_text(metrics_text + "\n", encoding="utf-8")
        mlflow.log_artifact(str(METRICS_PATH))

    result = {
        "model_path": str(MODEL_PATH),
        "metrics_path": str(METRICS_PATH),
        **metrics,
    }
    print("Model engineering complete:", result)
    return result


if __name__ == "__main__":
    run_model_engineering()
