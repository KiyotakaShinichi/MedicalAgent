from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.sim_to_public_imaging_report import DEFAULT_OUTPUT_PATH, build_sim_to_public_imaging_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build synthetic-to-public imaging gap report.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    result = build_sim_to_public_imaging_report(output_path=args.output)
    print(json.dumps({
        "status": result["status"],
        "synthetic_rows": result["synthetic_summary"].get("row_count"),
        "available_public_datasets": result["public_imaging_availability"].get("available_dataset_count"),
        "output_path": args.output,
    }, indent=2))


if __name__ == "__main__":
    main()
