from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.unsafe_intent_semantic_classifier import evaluate_unsafe_intent_classifier  # noqa: E402


def main() -> int:
    payload = evaluate_unsafe_intent_classifier()
    print(json.dumps({
        "status": payload["status"],
        "total_n": payload["total_n"],
        "pass_rate": payload["pass_rate"],
        "fail_count": payload["fail_count"],
    }, indent=2))
    return 0 if payload["status"] in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
