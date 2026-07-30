from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.xai_reliability_gate import build_xai_reliability_gate


def main() -> int:
    payload = build_xai_reliability_gate()
    print(
        json.dumps(
            {
                "status": payload["status"],
                "patient_display_policy": payload["patient_display_policy"],
                "promotion_blockers": payload["promotion_blockers"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
