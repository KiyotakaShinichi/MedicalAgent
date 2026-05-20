from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.priority_dataset_bridge import (  # noqa: E402
    DEFAULT_DOC_PATH,
    DEFAULT_OUTPUT_PATH,
    build_priority_dataset_bridge,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build GENIE BPC BRCA and Duke Breast MRI canonical bridge/readiness artifacts."
    )
    parser.add_argument("--genie-csv", default=None, help="Optional local GENIE BPC BRCA CSV export.")
    parser.add_argument("--duke-csv", default=None, help="Optional local Duke Breast MRI metadata CSV export.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--doc", default=DEFAULT_DOC_PATH)
    args = parser.parse_args()

    report = build_priority_dataset_bridge(
        genie_csv=args.genie_csv,
        duke_csv=args.duke_csv,
        output_path=args.output,
        doc_path=args.doc,
    )
    print(json.dumps({
        "status": report["status"],
        "mapped_dataset_count": report["summary"]["mapped_dataset_count"],
        "ready_for_mapping_count": report["summary"]["ready_for_mapping_count"],
        "highest_priority_next": report["summary"]["highest_priority_next"],
    }, indent=2))
    return 0 if report["status"] in {"strong", "ready_for_mapping"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
