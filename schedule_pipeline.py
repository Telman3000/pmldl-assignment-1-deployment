"""
Lightweight local scheduler (alternative to Airflow UI).
Runs the full pipeline every 5 minutes.
Use this if you prefer not to start the Airflow Docker stack.
"""

import time

import schedule

from run_pipeline import main as run_once


def job() -> None:
    print("\n--- Scheduled pipeline run starting ---")
    try:
        run_once()
    except Exception as exc:  # noqa: BLE001 - show error and keep scheduler alive
        print(f"Pipeline run failed: {exc}")


if __name__ == "__main__":
    job()  # run immediately once
    schedule.every(5).minutes.do(job)
    print("Scheduler started: pipeline every 5 minutes. Press Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(1)
