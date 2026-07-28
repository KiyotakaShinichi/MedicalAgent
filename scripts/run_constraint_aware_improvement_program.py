from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.constraint_aware_improvement_program import build_improvement_program


if __name__ == "__main__":
    report = build_improvement_program()
    print(json.dumps({
        "status": report["status"],
        "domain_count": report["domain_count"],
        "engineering_release_decision": report["engineering_release_decision"],
    }, indent=2))
