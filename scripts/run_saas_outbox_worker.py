from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.database import SessionLocal  # noqa: E402
from backend.services.saas_outbox_dispatcher import run_outbox_once  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Deliver redacted SaaS outbox events to signed n8n workflows.")
    parser.add_argument("--once", action="store_true", help="Process at most one available event.")
    parser.add_argument("--poll-seconds", type=float, default=3.0)
    args = parser.parse_args()
    worker_id = f"saas-outbox:{os.getpid()}"
    while True:
        db = SessionLocal()
        try:
            result = run_outbox_once(db, worker_id=worker_id)
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
