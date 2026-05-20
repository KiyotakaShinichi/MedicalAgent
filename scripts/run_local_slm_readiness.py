from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.local_slm_readiness import DEFAULT_OUTPUT_PATH, build_local_slm_readiness_manifest


def main() -> int:
    manifest = build_local_slm_readiness_manifest(output_path=DEFAULT_OUTPUT_PATH)
    print(json.dumps({
        "status": manifest.get("status"),
        "enabled_low_risk_tasks": len(manifest.get("enabled_low_risk_tasks") or []),
        "blocked_solo_tasks": len(manifest.get("blocked_solo_tasks") or []),
        "artifact": DEFAULT_OUTPUT_PATH,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
