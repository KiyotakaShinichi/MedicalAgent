from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.public_imaging_datasets import DEFAULT_OUTPUT_PATH, build_public_imaging_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local public imaging datasets and write a manifest.")
    parser.add_argument("--dataset-root", default="Datasets")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    result = build_public_imaging_manifest(dataset_root=args.dataset_root, output_path=args.output)
    print(json.dumps({
        "status": result["status"],
        "available_dataset_count": result["available_dataset_count"],
        "manifest_hash": result["manifest_hash"],
        "output_path": args.output,
    }, indent=2))


if __name__ == "__main__":
    main()
