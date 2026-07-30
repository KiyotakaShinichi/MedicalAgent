from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.provider_usage_reconciliation import (
    write_provider_usage_reconciliation,
)


if __name__ == "__main__":
    result = write_provider_usage_reconciliation()
    print(
        json.dumps(
            {
                "status": result["status"],
                "completed": result["completed"],
                "paired_request_count": result["paired_request_count"],
                "actual_usage_coverage_rate": result["actual_usage_coverage_rate"],
            },
            indent=2,
        )
    )
