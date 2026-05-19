from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.tcga_metabric_canonical_mapping import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    build_tcga_metabric_canonical_mapping,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build TCGA/METABRIC canonical mapping artifact.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--live-fetch", action="store_true")
    args = parser.parse_args()

    report = build_tcga_metabric_canonical_mapping(output_path=args.output, live_fetch=args.live_fetch)
    print(json.dumps({
        "status": report["status"],
        "mapped_dataset_count": report["mapped_dataset_count"],
        "output_path": args.output,
    }, indent=2))
    return 0 if report["status"] in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
