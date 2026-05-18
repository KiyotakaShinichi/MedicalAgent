import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.per_head_calibration import run_per_head_calibration


if __name__ == "__main__":
    payload = run_per_head_calibration()
    print(json.dumps({"status": payload["status"], "heads": list(payload["heads"].keys())}, indent=2))
    sys.exit(0 if payload["status"] == "strong" else 1)
