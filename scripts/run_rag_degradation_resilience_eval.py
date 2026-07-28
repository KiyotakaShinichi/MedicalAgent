from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.rag_degradation_resilience_eval import (  # noqa: E402
    build_rag_degradation_resilience_eval,
)


if __name__ == "__main__":
    payload = build_rag_degradation_resilience_eval()
    print(json.dumps({
        "status": payload["status"],
        "passed": payload["passed_count"],
        "total": payload["case_count"],
    }, indent=2))
