from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.live_rag_failure_analysis import build_live_rag_failure_analysis


def main() -> int:
    payload = build_live_rag_failure_analysis()
    print(json.dumps(payload.get("summary", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

