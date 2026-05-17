"""Run synthetic robustness stress suite."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.robustness_stress import run_robustness_stress_suite


if __name__ == "__main__":
    payload = run_robustness_stress_suite()
    print(json.dumps(payload.get("summary", {}), indent=2))
    sys.exit(0 if payload.get("status") in {"strong", "acceptable"} else 1)
