from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.database import SessionLocal  # noqa: E402
from backend.services.saas_job_worker import run_platform_job_once  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the synthetic SaaS evaluation worker.")
    parser.add_argument("--once", action="store_true", help="Process at most one available job.")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()
    worker_id = f"saas-worker:{os.getpid()}"
    while True:
        db = SessionLocal()
        try:
            result = run_platform_job_once(db, worker_id=worker_id)
        finally:
            db.close()
        if result is not None:
            print(f"{result['id']} {result['status']}", flush=True)
        if args.once:
            return 0
        if result is None:
            time.sleep(max(0.25, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
