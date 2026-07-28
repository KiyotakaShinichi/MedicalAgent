from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.agent_execution_policy_eval import (  # noqa: E402
    build_agent_execution_policy_eval,
)


if __name__ == "__main__":
    result = build_agent_execution_policy_eval()
    print(json.dumps({
        "status": result["status"],
        "passed": result["passed_count"],
        "total": result["case_count"],
        "live_patient_write_performed": result["live_patient_write_performed"],
    }, indent=2))
