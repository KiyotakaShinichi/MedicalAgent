from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.toxicity_review_target_v3 import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    run_toxicity_review_target_v3,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the synthetic toxicity review target v3 artifact.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    report = run_toxicity_review_target_v3(output_path=args.output)
    print(f"wrote: {args.output}")
    print(json.dumps({
        "status": report["status"],
        "model_auroc": report["model"]["auroc"],
        "legacy_rule_auroc_against_v3": report["shortcut_comparison"]["legacy_rule_auroc_against_v3"],
        "promotion_decision": report["recommendation"]["promotion_decision"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
