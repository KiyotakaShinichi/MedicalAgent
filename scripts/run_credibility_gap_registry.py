from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.credibility_gap_registry import build_credibility_gap_registry


if __name__ == "__main__":
    report = build_credibility_gap_registry()
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, indent=2))
