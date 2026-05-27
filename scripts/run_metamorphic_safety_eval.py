from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.metamorphic_safety_eval import evaluate_metamorphic_safety  # noqa: E402


def main() -> int:
    payload = evaluate_metamorphic_safety()
    print(json.dumps({
        "status": payload["status"],
        "total_n": payload["total_n"],
        "pass_rate": payload["pass_rate"],
        "unsafe_route_preservation_rate": payload["unsafe_route_preservation_rate"],
        "safe_negative_preservation_rate": payload["safe_negative_preservation_rate"],
        "unsafe_write_leakage_count": payload["unsafe_write_leakage_count"],
    }, indent=2))
    return 0 if payload["status"] in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
