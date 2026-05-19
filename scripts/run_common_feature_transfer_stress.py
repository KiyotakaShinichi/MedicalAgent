from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.common_feature_transfer_stress import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    run_common_feature_transfer_stress,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run common-feature transfer stress across synthetic/public cohorts.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = run_common_feature_transfer_stress(output_path=args.output)
    print(json.dumps({
        "status": report["status"],
        "cohort_sizes": report["cohort_sizes"],
        "promotion_allowed": report["promotion_decision"]["promotion_allowed"],
        "warning_count": len(report["warnings"]),
        "output_path": args.output,
    }, indent=2))
    return 0 if report["status"] in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
