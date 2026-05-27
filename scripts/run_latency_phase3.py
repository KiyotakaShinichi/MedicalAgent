from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.latency_phase3 import build_latency_phase3_plan  # noqa: E402


def main() -> int:
    payload = build_latency_phase3_plan()
    print(json.dumps({
        "status": payload["status"],
        "headline_metric": payload["headline_metric"],
        "production_ready": payload["production_ready"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
