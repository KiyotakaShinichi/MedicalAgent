from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.priority_external_stress import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    build_priority_external_stress,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run external schema/endpoint stress over priority bridge rows.")
    parser.add_argument("--bridge", default="Data/evals/models/latest_priority_dataset_bridge.json")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    report = build_priority_external_stress(bridge_path=args.bridge, output_path=args.output)
    print(json.dumps({
        "status": report["status"],
        "mapped_dataset_count": report["mapped_dataset_count"],
        "promotion_allowed": report["promotion_decision"]["promotion_allowed"],
    }, indent=2))
    return 0 if report["promotion_decision"]["promotion_allowed"] is False else 1


if __name__ == "__main__":
    raise SystemExit(main())
