from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.synthetic_load_matrix import run_synthetic_load_matrix


if __name__ == "__main__":
    result = run_synthetic_load_matrix()
    print(json.dumps({"status": result["status"], "profiles": result["profiles"]}, indent=2))
