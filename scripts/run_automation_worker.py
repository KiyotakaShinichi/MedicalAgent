import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.automation_worker import default_worker_id, run_automation_worker_once
from backend.schema_migrations import ensure_schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the durable redacted-engineering automation worker.")
    parser.add_argument("--once", action="store_true", help="Process at most one job, then exit.")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--lease-seconds", type=int, default=120)
    parser.add_argument("--worker-id", default=default_worker_id())
    args = parser.parse_args()
    ensure_schema()
    while True:
        result = run_automation_worker_once(worker_id=args.worker_id, lease_seconds=args.lease_seconds)
        if result is not None:
            print(f"task_id={result['id']} status={result['status']} attempts={result['attempts']}")
        if args.once:
            return
        if result is None:
            time.sleep(max(0.2, args.poll_seconds))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
