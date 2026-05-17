from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.cbioportal_biomarker_mapper import build_cbioportal_biomarker_schema_mapping


if __name__ == "__main__":
    report = build_cbioportal_biomarker_schema_mapping()
    print(json.dumps({
        "status": report.get("status"),
        "mapped_dataset_count": report.get("mapped_dataset_count"),
        "mapping_hash": report.get("mapping_hash"),
    }, indent=2))
