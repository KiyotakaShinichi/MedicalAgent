from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.rag_failure_attribution_vnext import build_rag_failure_attribution


if __name__ == "__main__":
    result = build_rag_failure_attribution()
    print(json.dumps({
        "status": result["status"],
        "failure_row_count": result["failure_row_count"],
        "aggregate_by_stage": result["aggregate_by_stage"],
    }, indent=2))
