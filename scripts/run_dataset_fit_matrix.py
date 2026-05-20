from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.dataset_fit_matrix import (  # noqa: E402
    DEFAULT_DOC_PATH,
    DEFAULT_OUTPUT_PATH,
    build_dataset_fit_matrix,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dataset fit/prioritization matrix.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--doc", default=DEFAULT_DOC_PATH)
    args = parser.parse_args()
    report = build_dataset_fit_matrix(output_path=args.output, doc_path=args.doc)
    print(json.dumps({
        "status": report["status"],
        "dataset_count": report["dataset_count"],
        "top_5": report["top_5"],
        "production_training_allowed": report["recommendation"]["production_training_allowed"],
    }, indent=2))
    return 0 if report["recommendation"]["production_training_allowed"] is False else 1


if __name__ == "__main__":
    raise SystemExit(main())
