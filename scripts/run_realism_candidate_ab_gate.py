from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.realism_candidate_ab_gate import DEFAULT_OUTPUT_PATH, run_realism_candidate_ab_gate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run current-vs-public-realism-candidate A/B gate.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = run_realism_candidate_ab_gate(output_path=args.output)
    print(json.dumps({
        "status": report["status"],
        "decision": report["recommendation"]["decision"],
        "candidate_use": report["recommendation"]["candidate_use"],
        "classification_auroc_delta": report["deltas"]["classification_auroc_delta"],
        "regression_mae_delta": report["deltas"]["regression_mae_delta"],
        "production_replacement_allowed": report["recommendation"]["production_replacement_allowed"],
        "output_path": args.output,
    }, indent=2))
    return 0 if report["status"] in {"candidate", "strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
