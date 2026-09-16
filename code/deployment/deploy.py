"""
Stage 3 — Deployment helpers
Build and (re)start API + Streamlit app via Docker Compose.

The model is baked into the API image at build time (see api/Dockerfile).
No host bind-mount for models — that breaks DooD on Docker Desktop/Windows.
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = PROJECT_ROOT / "code" / "deployment" / "docker-compose.yml"
MODEL_PATH = PROJECT_ROOT / "models" / "titanic_model.joblib"
AIRFLOW_MOUNT = Path("/opt/project")


def _as_posix(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def _build_context() -> str:
    """
    Build context path for the Compose *client* filesystem.

    Inside Airflow the repo is mounted at /opt/project — use that so the
    client can tar the context (including models/ for COPY into the image).
    On the host, use the real project root.
    """
    if (AIRFLOW_MOUNT / "code" / "deployment" / "docker-compose.yml").exists():
        return "/opt/project"
    return _as_posix(PROJECT_ROOT.resolve())


def run_deployment() -> dict:
    model_ok = MODEL_PATH.exists() or (
        AIRFLOW_MOUNT / "models" / "titanic_model.joblib"
    ).exists()
    if not model_ok:
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run model engineering first."
        )
    if not COMPOSE_FILE.exists():
        raise FileNotFoundError(f"docker-compose.yml not found at {COMPOSE_FILE}")

    build_context = _build_context()
    env = os.environ.copy()
    env["COMPOSE_BUILD_CONTEXT"] = build_context

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
    print("COMPOSE_BUILD_CONTEXT=", build_context)
    completed = subprocess.run(
        cmd, check=True, capture_output=True, text=True, env=env
    )

    result = {
        "compose_file": str(COMPOSE_FILE),
        "compose_build_context": build_context,
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
