from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.release_gate_explanation import write_release_gate_explanation


def main() -> int:
    payload = write_release_gate_explanation()
    print(json.dumps({"status": payload.get("status")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

