from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.synthetic_staging_runtime_drill import (  # noqa: E402
    inside_task_phase,
    run_runtime_recovery_drill,
    status_by_task_id,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("Data/evals/ops/latest_synthetic_staging_runtime_recovery.json"))
    parser.add_argument("--inside-phase", choices=("prepare", "status", "replay", "cleanup"))
    parser.add_argument("--drill-id")
    parser.add_argument("--status-by-task-id", action="store_true")
    args = parser.parse_args()
    if args.inside_phase:
        if not args.drill_id:
            parser.error("--drill-id is required for an inside phase")
        if args.status_by_task_id:
            payload = status_by_task_id(int(args.drill_id))
        else:
            payload = inside_task_phase(args.inside_phase, args.drill_id)
        print(json.dumps(payload))
        return 0
    payload = run_runtime_recovery_drill(output_path=args.output)
    print(json.dumps({
        "status": payload["status"],
        "passed_count": payload["passed_count"],
        "check_count": payload["check_count"],
        "worker_running_after_drill": payload["worker_running_after_drill"],
        "error": payload["error"],
    }, indent=2))
    return 0 if payload["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
