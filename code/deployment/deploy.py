"""
Stage 3 — Deployment helpers
Build and (re)start API + Streamlit app via Docker Compose.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = PROJECT_ROOT / "code" / "deployment" / "docker-compose.yml"
MODEL_PATH = PROJECT_ROOT / "models" / "titanic_model.joblib"


def run_deployment() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run model engineering first."
        )
    if not COMPOSE_FILE.exists():
        raise FileNotFoundError(f"docker-compose.yml not found at {COMPOSE_FILE}")

    cmd = [
        "docker",
        "compose",
        "-f",
        str(COMPOSE_FILE),
        "up",
        "--build",
        "-d",
        "--force-recreate",
    ]
    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)

    result = {
        "compose_file": str(COMPOSE_FILE),
        "api_url": "http://localhost:8000",
        "app_url": "http://localhost:8501",
        "stdout": completed.stdout[-1000:],
    }
    print("Deployment complete:", {k: v for k, v in result.items() if k != "stdout"})
    return result


if __name__ == "__main__":
    try:
        run_deployment()
    except subprocess.CalledProcessError as exc:
        print(exc.stdout)
        print(exc.stderr, file=sys.stderr)
        raise
