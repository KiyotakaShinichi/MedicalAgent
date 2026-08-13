from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.dep001a_candidate_freeze import build_freeze_manifest


if __name__ == "__main__":
    print(json.dumps(build_freeze_manifest(), indent=2))
