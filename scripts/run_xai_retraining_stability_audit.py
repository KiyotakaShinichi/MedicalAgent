from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.xai_retraining_stability_audit import (  # noqa: E402
    build_xai_retraining_stability_audit,
)


if __name__ == "__main__":
    payload = build_xai_retraining_stability_audit()
    print(json.dumps({
        "status": payload["status"],
        "seed_count": payload["seed_count"],
        "metrics": payload["metrics"],
    }, indent=2))
