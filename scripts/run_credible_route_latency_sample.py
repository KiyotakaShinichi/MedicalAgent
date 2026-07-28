from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.credible_route_latency_sample import (
    build_credible_route_latency_sample,
)


def main() -> int:
    payload = build_credible_route_latency_sample()
    print(
        json.dumps(
            {
                "status": payload["status"],
                "routes": [
                    {
                        "route": row["route"],
                        "sample_count": row["sample_count"],
                        "p95_ms": row["current_p95_ms"],
                        "latency_status": row["latency_status"],
                    }
                    for row in payload["routes"]
                ],
                "production_ready": payload["production_ready"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
