from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.ml_statistical_robustness import build_ml_statistical_robustness  # noqa: E402


def main() -> int:
    payload = build_ml_statistical_robustness()
    print(json.dumps({
        "status": payload["status"],
        "total_n": payload["total_n"],
        "classification_accuracy": payload["classification_bootstrap"]["observed"]["accuracy"],
        "regression_mae": payload["regression_bootstrap"]["observed"]["mae"],
        "flag_count": len(payload["stability_flags"]),
    }, indent=2))
    return 0 if payload["status"] in {"acceptable", "strong"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
