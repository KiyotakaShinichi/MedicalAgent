from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.external_dataset_integration_matrix import (  # noqa: E402
    DEFAULT_DOC_PATH,
    DEFAULT_OUTPUT_PATH,
    build_external_dataset_integration_matrix,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the external dataset integration matrix.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--doc", default=DEFAULT_DOC_PATH)
    args = parser.parse_args()

    report = build_external_dataset_integration_matrix(output_path=args.output, doc_path=args.doc)
    print(
        json.dumps(
            {
                "status": report["status"],
                "dataset_count": report["dataset_count"],
                "clinical_validation": report["clinical_validation"],
                "production_training_allowed": report["production_training_allowed"],
                "top_next_integrations": [
                    item["dataset_id"] for item in report["highest_roi_next_integrations"]
                ],
                "output_path": args.output,
                "doc_path": args.doc,
            },
            indent=2,
        )
    )
    return 0 if report["status"] in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
