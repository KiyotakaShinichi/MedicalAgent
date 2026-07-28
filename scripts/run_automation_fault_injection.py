from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.automation_fault_injection_eval import build_automation_fault_injection_eval  # noqa: E402


if __name__ == "__main__":
    result = build_automation_fault_injection_eval()
    print(json.dumps({
        "status": result["status"],
        "scenario_count": result["scenario_count"],
        "passed_count": result["passed_count"],
    }, indent=2))
