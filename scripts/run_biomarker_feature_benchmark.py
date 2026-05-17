from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.biomarker_feature_benchmark import run_biomarker_feature_benchmark


def main() -> None:
    result = run_biomarker_feature_benchmark()
    summary = {
        "status": result.get("status"),
        "patients": result.get("patients"),
        "test_patients": result.get("test_patients"),
        "deltas": result.get("deltas"),
        "recommendation": result.get("recommendation"),
        "output_path": result.get("artifacts", {}).get("report_json"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
