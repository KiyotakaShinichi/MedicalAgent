from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.adversarial_v6_tuning_regression import (
    build_v6_tuning_regression,
)


def main() -> int:
    payload = build_v6_tuning_regression()
    print(
        json.dumps(
            {
                "status": payload["status"],
                "baseline_pass_rate": payload["baseline_pass_rate"],
                "tuning_regression_pass_rate": payload["tuning_regression_pass_rate"],
                "pass_rate_delta": payload["pass_rate_delta"],
                "independent_holdout_evidence": payload[
                    "independent_holdout_evidence"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
