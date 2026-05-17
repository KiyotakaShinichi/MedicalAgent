"""Build response-score conformal interval calibration artifact."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.response_conformal_calibration import build_response_conformal_calibration


if __name__ == "__main__":
    payload = build_response_conformal_calibration()
    print(json.dumps({
        "status": payload.get("status"),
        "nominal_coverage": payload.get("nominal_coverage"),
        "raw_coverage": payload.get("raw_coverage"),
        "adjusted_coverage": payload.get("adjusted_coverage"),
        "qhat_percent": payload.get("qhat_percent"),
        "calibration_rows": payload.get("calibration_rows"),
    }, indent=2))
    sys.exit(0 if payload.get("status") in {"strong", "acceptable"} else 1)
