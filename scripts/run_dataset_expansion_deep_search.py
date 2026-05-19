from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.dataset_expansion_deep_search import (  # noqa: E402
    DEFAULT_DOC_PATH,
    DEFAULT_OUTPUT_PATH,
    build_dataset_expansion_deep_search,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build governed dataset-expansion deep-search artifact.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--doc", default=DEFAULT_DOC_PATH)
    args = parser.parse_args()

    report = build_dataset_expansion_deep_search(output_path=args.output, doc_path=args.doc)
    print(json.dumps({
        "status": report["status"],
        "dataset_count": report["dataset_count"],
        "highest_priority": [row["id"] for row in report["highest_priority"]],
        "output_path": args.output,
        "doc_path": args.doc,
    }, indent=2))
    return 0 if report["status"] in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
