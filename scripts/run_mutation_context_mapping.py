from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.mutation_context_mapping import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    build_mutation_context_mapping,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build mutation-context mapping readiness artifact.")
    parser.add_argument("--mutation-csv", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    report = build_mutation_context_mapping(mutation_csv=args.mutation_csv, output_path=args.output)
    print(json.dumps({
        "status": report["status"],
        "mapped_row_count": report["mapped_row_count"],
        "promotion_allowed": report["feature_policy"]["promotion_allowed"],
    }, indent=2))
    return 0 if report["feature_policy"]["promotion_allowed"] is False else 1


if __name__ == "__main__":
    raise SystemExit(main())
