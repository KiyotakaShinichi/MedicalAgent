from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.cbioportal_clinical_export import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    build_cbioportal_clinical_export,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export public cBioPortal clinical rows to canonical NLCare schema.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--offline", action="store_true", help="Skip live cBioPortal fetch.")
    args = parser.parse_args()

    report = build_cbioportal_clinical_export(output_path=args.output, live_fetch=not args.offline)
    print(json.dumps({
        "status": report["status"],
        "row_count": report["combined"]["row_count"],
        "studies": {
            key: {"status": value["status"], "rows": value["row_count"]}
            for key, value in report["studies"].items()
        },
        "output_path": args.output,
    }, indent=2))
    return 0 if report["status"] in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
