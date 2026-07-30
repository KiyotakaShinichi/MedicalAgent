from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.cross_domain_assurance_eval import (
    build_cross_domain_assurance_eval,
)


if __name__ == "__main__":
    report = build_cross_domain_assurance_eval()
    print(
        json.dumps(
            {
                "status": report["status"],
                "passed_count": report["passed_count"],
                "scenario_count": report["scenario_count"],
                "external_network_request_performed": report[
                    "external_network_request_performed"
                ],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["failed_count"] == 0 else 1)
