from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.adversarial_hardening_report import build_adversarial_hardening_report  # noqa: E402


def main() -> int:
    payload = build_adversarial_hardening_report()
    print(json.dumps({
        "status": payload["status"],
        "v3_pass_rate_delta": payload["v3_delta"]["pass_rate_delta"],
        "v4_pass_rate": payload["v4_fresh_holdout"].get("pass_rate"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
