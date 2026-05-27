from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.agentic_shadow_mode import build_agentic_shadow_mode_eval  # noqa: E402


def main() -> int:
    payload = build_agentic_shadow_mode_eval()
    print(json.dumps({
        "status": payload["status"],
        "total_n": payload["total_n"],
        "route_match_rate": payload["route_match_rate"],
        "unsafe_write_leakage_count": payload["unsafe_write_leakage_count"],
    }, indent=2))
    return 0 if payload["status"] in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
