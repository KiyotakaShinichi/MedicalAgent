from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.external_data_bridge import (  # noqa: E402
    DEFAULT_CANONICAL_CSV,
    DEFAULT_FAILURE_GALLERY_PATH,
    DEFAULT_OUTPUT_PATH,
    build_external_data_bridge,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build canonical external-data bridge artifacts.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--canonical-csv", default=DEFAULT_CANONICAL_CSV)
    parser.add_argument("--failure-gallery", default=DEFAULT_FAILURE_GALLERY_PATH)
    args = parser.parse_args()

    report = build_external_data_bridge(
        output_path=args.output,
        canonical_csv=args.canonical_csv,
        failure_gallery_path=args.failure_gallery,
    )
    print(json.dumps({
        "status": report["status"],
        "row_count": report["row_count"],
        "validation_status": report["validation"]["status"],
        "failure_cases": report["failure_gallery_summary"]["case_count"],
        "output_path": args.output,
    }, indent=2))
    return 0 if report["status"] in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
