from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.row_level_prediction_export import run_row_level_prediction_evidence  # noqa: E402


def main() -> int:
    payload = run_row_level_prediction_evidence()
    manifest = payload["manifest"]
    paired = payload["paired"]
    calibration = payload["calibration"]
    print(json.dumps({
        "manifest_status": manifest["status"],
        "total_n": manifest["total_n"],
        "patient_id_unique": manifest["patient_id_unique"],
        "paired_status": paired["status"],
        "calibration_status": calibration["status"],
        "max_ece": calibration["max_ece"],
    }, indent=2))
    return 0 if manifest["status"] in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
