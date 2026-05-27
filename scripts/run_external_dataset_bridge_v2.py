from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.external_dataset_bridge_v2 import build_external_dataset_bridge_v2  # noqa: E402


def main() -> int:
    payload = build_external_dataset_bridge_v2()
    print(json.dumps({
        "status": payload["status"],
        "highest_priority": payload["highest_priority"],
        "clinical_validation": payload["clinical_validation"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
