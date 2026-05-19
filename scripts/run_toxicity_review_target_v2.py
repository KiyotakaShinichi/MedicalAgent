from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.toxicity_review_target_v2 import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    run_toxicity_review_target_v2,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run toxicity review target v2 benchmark.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = run_toxicity_review_target_v2(output_path=args.output)
    print(json.dumps({
        "status": report["status"],
        "auroc": report["model"]["auroc"],
        "legacy_rule_accuracy_against_v2": report["shortcut_comparison"]["legacy_rule_accuracy_against_v2"],
        "output_path": args.output,
    }, indent=2))
    return 0 if report["status"] in {"candidate", "strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
