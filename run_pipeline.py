"""
Run the full MLOps pipeline once:
1) data engineering  2) model engineering  3) deployment
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code" / "datasets"))
sys.path.insert(0, str(ROOT / "code" / "models"))
sys.path.insert(0, str(ROOT / "code" / "deployment"))

from data_engineering import run_data_engineering  # noqa: E402
from deploy import run_deployment  # noqa: E402
from model_engineering import run_model_engineering  # noqa: E402


def main() -> None:
    print("=== Stage 1: Data Engineering ===")
    run_data_engineering()

    print("=== Stage 2: Model Engineering ===")
    run_model_engineering()

    print("=== Stage 3: Deployment ===")
    run_deployment()

    print("Pipeline finished.")
    print("API:  http://localhost:8000/docs")
    print("App:  http://localhost:8501")


if __name__ == "__main__":
    main()
