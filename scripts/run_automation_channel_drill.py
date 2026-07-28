from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.automation_channel_drill import (
    build_automation_channel_drill,
)


def main() -> int:
    payload = build_automation_channel_drill()
    print(
        json.dumps(
            {
                "status": payload["status"],
                "attempt_count": payload["attempt_count"],
                "pass_rate": payload["pass_rate"],
                "latency_ms": payload["latency_ms"],
                "external_delivery_performed": payload[
                    "external_delivery_performed"
                ],
            },
            indent=2,
        )
    )
    return 0 if payload["status"] == "strong" else 1


if __name__ == "__main__":
    raise SystemExit(main())
