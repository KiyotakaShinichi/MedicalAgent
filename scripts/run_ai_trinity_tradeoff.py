from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.ai_trinity_tradeoff import write_ai_trinity_tradeoff


if __name__ == "__main__":
    artifact = write_ai_trinity_tradeoff()
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "decision": artifact["decision"],
                "promotion_allowed": artifact["promotion_allowed"],
                "axes": artifact["axes"],
            },
            indent=2,
        )
    )
