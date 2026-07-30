from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.evidence_maturity_matrix import (
    build_evidence_maturity_matrix,
)


def main() -> int:
    payload = build_evidence_maturity_matrix()
    print(
        json.dumps(
            {
                "status": payload["status"],
                "tier_counts": payload["scoring_policy"]["tier_counts"],
                "architecture": payload["architecture_maintainability"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
