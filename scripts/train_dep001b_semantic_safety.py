from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.dep001b_semantic_safety_training import train_and_evaluate


if __name__ == "__main__":
    print(json.dumps(train_and_evaluate(), indent=2))
