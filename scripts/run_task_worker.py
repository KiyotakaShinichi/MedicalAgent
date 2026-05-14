"""Run queued admin jobs from the local database.

This is intentionally small and account-free: Docker Compose can run it as a
background worker, while local demos can run one job with --once.
"""

from __future__ import annotations

import argparse
import time

from backend.database import SessionLocal
from backend.schema_migrations import ensure_schema
from backend.services.task_queue import run_next_queued_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MedicalAgent async tasks.")
    parser.add_argument("--once", action="store_true", help="Process one queued task and exit.")
    parser.add_argument("--poll-seconds", type=float, default=5.0, help="Sleep time when no task is queued.")
    args = parser.parse_args()

    ensure_schema()
    while True:
        db = SessionLocal()
        try:
            task = run_next_queued_task(db)
            if task:
                print(f"completed task {task['id']} ({task['task_type']}): {task['status']}", flush=True)
            elif args.once:
                print("no queued tasks", flush=True)
        finally:
            db.close()

        if args.once:
            break
        if task is None:
            time.sleep(max(args.poll_seconds, 0.5))


if __name__ == "__main__":
    main()
