from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

from backend.services.dep001a_runtime_assurance import build_dep001a_runtime_assurance


if __name__ == "__main__":
    result = build_dep001a_runtime_assurance()
    print(json.dumps({
        "status": result["status"],
        "dep001_status": result["dep001_status"],
        "metrics": result["metrics"],
        "fault_injection": {
            "passed": result["fault_injection"]["passed"],
            "passed_n": result["fault_injection"]["passed_n"],
            "total_n": result["fault_injection"]["total_n"],
        },
    }, indent=2))
