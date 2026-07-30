from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.senior_engineering_evidence import (
    build_senior_engineering_evidence,
)


if __name__ == "__main__":
    report = build_senior_engineering_evidence()
    print(
        json.dumps(
            {
                "status": report["status"],
                "evidence_maturity": report["evidence_maturity"],
                "architecture_fitness_pass_rate": report[
                    "architecture_fitness_pass_rate"
                ],
                "independent_reproduction_completed": report[
                    "independent_reproduction_completed"
                ],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["status"] != "needs_attention" else 1)
