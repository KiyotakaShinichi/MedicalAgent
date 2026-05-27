from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.adversarial_holdout_v3 import evaluate_holdout_v3  # noqa: E402


def main() -> int:
    payload = evaluate_holdout_v3()
    print(json.dumps({
        "status": payload["status"],
        "total_n": payload["total_n"],
        "pass_rate": payload["pass_rate"],
        "unsafe_leakage_rate": payload["unsafe_leakage_rate"],
        "over_refusal_rate": payload["over_refusal_rate"],
    }, indent=2))
    # This is a warning baseline, not a hard failure while newly frozen.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
