from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.synthetic_model_perturbation_retrain_eval import (
    write_synthetic_model_perturbation_retrain_eval,
)


if __name__ == "__main__":
    result = write_synthetic_model_perturbation_retrain_eval()
    print(
        json.dumps(
            {
                "status": result["status"],
                "stress_failure_count": len(result["stress_failures"]),
                "promotion_decision": result["promotion_decision"],
            },
            indent=2,
        )
    )
