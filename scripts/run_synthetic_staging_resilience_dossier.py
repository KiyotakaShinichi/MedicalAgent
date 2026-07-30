from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.synthetic_staging_resilience_dossier import (
    write_synthetic_staging_resilience_dossier,
)


if __name__ == "__main__":
    result = write_synthetic_staging_resilience_dossier()
    print(
        json.dumps(
            {
                "status": result["status"],
                "local_checks": (
                    f"{result['local_passed_count']}/{result['local_check_count']}"
                ),
                "unresolved_external_or_managed_blockers": result[
                    "unresolved_external_or_managed_blocker_count"
                ],
            },
            indent=2,
        )
    )
