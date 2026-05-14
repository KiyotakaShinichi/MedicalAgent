from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.imaging_baseline_experiments import DEFAULT_CT_WORKFLOW_PATH, build_ct_lesion_workflow_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CT/PET-CT lesion workflow-readiness report.")
    parser.add_argument("--dataset-root", default="Datasets/DeepLesion")
    parser.add_argument("--output", default=DEFAULT_CT_WORKFLOW_PATH)
    args = parser.parse_args()

    result = build_ct_lesion_workflow_report(dataset_root=args.dataset_root, output_path=args.output)
    print(json.dumps({
        "status": result["status"],
        "image_file_count": result.get("image_file_count"),
        "metadata_file_count": result.get("metadata_file_count"),
        "output_path": args.output,
    }, indent=2))


if __name__ == "__main__":
    main()
