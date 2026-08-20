from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.dep001b_safety_corpus import build_corpora


if __name__ == "__main__":
    print(json.dumps(build_corpora(), indent=2))
