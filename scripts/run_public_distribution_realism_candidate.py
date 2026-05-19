from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.public_distribution_realism_candidate import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    build_public_distribution_realism_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build public-distribution-tuned synthetic realism candidate.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = build_public_distribution_realism_candidate(output_path=args.output)
    print(json.dumps({
        "status": report["status"],
        "rows": report["rows"],
        "patients": report["patients"],
        "candidate_csv": report["candidate_csv"],
        "improved_fields": [
            key for key, value in report["before_after_gaps"].items()
            if value.get("gap_improved")
        ],
        "output_path": args.output,
    }, indent=2))
    return 0 if report["status"] in {"candidate", "strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
