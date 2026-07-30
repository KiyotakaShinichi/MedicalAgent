from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.finetune_hardening_assurance import (
    build_finetune_hardening_assurance,
)


if __name__ == "__main__":
    report = build_finetune_hardening_assurance()
    print(
        json.dumps(
            {
                "status": report["status"],
                "promotion_decision": report["promotion_decision"],
                "summary": report["summary"],
            },
            indent=2,
        )
    )
