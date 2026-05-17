from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.public_biomarker_mapping import build_public_biomarker_mapping_readiness


def main() -> None:
    result = build_public_biomarker_mapping_readiness()
    print(json.dumps({
        "status": result.get("status"),
        "mapping_hash": result.get("mapping_hash"),
        "breastdcedl": result.get("datasets", {}).get("breastdcedl", {}),
        "output_path": "Data/mle_monitoring/public_biomarker_mapping_readiness.json",
    }, indent=2))


if __name__ == "__main__":
    main()
