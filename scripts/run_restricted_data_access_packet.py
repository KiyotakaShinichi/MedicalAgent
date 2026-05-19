from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.restricted_data_access_packet import (  # noqa: E402
    DEFAULT_MD_PATH,
    DEFAULT_OUTPUT_PATH,
    build_restricted_data_access_packet,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build future restricted-dataset access packet.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--md", default=DEFAULT_MD_PATH)
    args = parser.parse_args()

    report = build_restricted_data_access_packet(output_path=args.output, md_path=args.md)
    print(json.dumps({
        "status": report["status"],
        "dataset_count": len(report["datasets"]),
        "output_path": args.output,
        "md_path": args.md,
    }, indent=2))
    return 0 if report["status"] == "ready_for_future_access_request" else 1


if __name__ == "__main__":
    raise SystemExit(main())
