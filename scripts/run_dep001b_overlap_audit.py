from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.dep001b_overlap_audit import run_overlap_audit


if __name__ == "__main__":
    print(json.dumps(run_overlap_audit(), indent=2))
