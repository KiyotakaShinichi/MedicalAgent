from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.production_readiness_boundary import build_production_readiness_boundary  # noqa: E402


def main() -> int:
    payload = build_production_readiness_boundary()
    print(json.dumps({
        "status": payload["status"],
        "healthcare_production_ready": payload["healthcare_production_ready"],
        "headline_metric": payload["headline_metric"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
