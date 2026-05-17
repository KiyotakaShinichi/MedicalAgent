from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.public_biomarker_datasets import build_public_biomarker_dataset_manifest


def main() -> None:
    result = build_public_biomarker_dataset_manifest()
    print(json.dumps({
        "status": result["status"],
        "dataset_count": result["dataset_count"],
        "recommended_order": result["recommended_order"],
        "output_path": "Data/data_lineage/public_biomarker_dataset_manifest.json",
    }, indent=2))


if __name__ == "__main__":
    main()
