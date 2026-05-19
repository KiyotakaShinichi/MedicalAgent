from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.external_distribution_alignment import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    build_external_distribution_alignment,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build external distribution alignment artifact.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = build_external_distribution_alignment(output_path=args.output)
    print(json.dumps({
        "status": report["status"],
        "cohort_sizes": report["cohort_sizes"],
        "largest_gap": report["largest_gaps"][0] if report["largest_gaps"] else None,
        "output_path": args.output,
    }, indent=2))
    return 0 if report["status"] in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
