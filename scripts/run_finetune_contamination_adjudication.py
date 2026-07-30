from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.finetune_contamination_adjudication import (
    write_finetune_contamination_adjudication,
)


if __name__ == "__main__":
    result = write_finetune_contamination_adjudication()
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_count": result["candidate_count"],
                "unresolved_count": result["unresolved_count"],
                "critical_unresolved_count": result["critical_unresolved_count"],
            },
            indent=2,
        )
    )
