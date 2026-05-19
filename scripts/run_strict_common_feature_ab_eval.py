from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.strict_common_feature_ab_eval import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    run_strict_common_feature_ab_eval,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run strict common-feature A/B eval.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = run_strict_common_feature_ab_eval(output_path=args.output)
    print(json.dumps({
        "status": report["status"],
        "synthetic_auc": report["datasets"]["synthetic_patient_level"]["metrics"].get("roc_auc"),
        "external_auc": report["datasets"]["breastdcedl_spy1"]["metrics"].get("roc_auc"),
        "decision": report["ab_decision"]["decision"],
        "output_path": args.output,
    }, indent=2))
    return 0 if report["status"] in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
