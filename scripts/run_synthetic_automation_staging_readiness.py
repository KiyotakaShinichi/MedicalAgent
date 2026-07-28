from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.synthetic_automation_staging_readiness import (  # noqa: E402
    build_synthetic_automation_staging_readiness,
)


if __name__ == "__main__":
    result = build_synthetic_automation_staging_readiness()
    print(json.dumps({
        "status": result["status"],
        "checks": f"{result['passed_count']}/{result['check_count']}",
        "compose_valid": result["compose_validation"]["valid"],
        "runtime_completed": result["runtime_completed"],
    }, indent=2))
