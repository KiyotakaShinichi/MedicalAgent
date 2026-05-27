from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.external_review_readiness import build_external_review_readiness  # noqa: E402


def main() -> int:
    payload = build_external_review_readiness()
    print(json.dumps({
        "status": payload["status"],
        "completed_external_review_count": payload["completed_external_review_count"],
        "headline_metric": payload["headline_metric"],
    }, indent=2))
    return 0 if payload["status"] in {"ready_for_external_authoring", "acceptable", "strong"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
