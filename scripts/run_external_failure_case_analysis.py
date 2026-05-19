from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.external_failure_case_analysis import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    build_external_failure_case_analysis,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build external failure-case subtype/confidence analysis.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = build_external_failure_case_analysis(output_path=args.output)
    print(json.dumps({
        "status": report["status"],
        "failure_count": report["summary"]["failure_count"],
        "high_confidence_failure_count": report["summary"]["high_confidence_failure_count"],
        "output_path": args.output,
    }, indent=2))
    return 0 if report["status"] in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
