from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.cost_latency_observability import DEFAULT_OUTPUT_PATH, build_cost_latency_report


def main() -> int:
    report = build_cost_latency_report(output_path=DEFAULT_OUTPUT_PATH)
    print(json.dumps({
        "status": report.get("status"),
        "request_count": (report.get("summary") or {}).get("request_count"),
        "p95_latency_ms": ((report.get("summary") or {}).get("overall_latency_ms") or {}).get("p95"),
        "artifact": DEFAULT_OUTPUT_PATH,
    }, indent=2))
    return 0 if report.get("status") in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
