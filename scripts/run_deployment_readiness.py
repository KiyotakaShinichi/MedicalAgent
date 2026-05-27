from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.deployment_readiness import build_deployment_readiness  # noqa: E402


def main() -> int:
    payload = build_deployment_readiness()
    print(json.dumps({
        "status": payload["status"],
        "headline_metric": payload["headline_metric"],
        "healthcare_production_ready": payload["healthcare_production_ready"],
        "clinical_validation": payload["clinical_validation"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
