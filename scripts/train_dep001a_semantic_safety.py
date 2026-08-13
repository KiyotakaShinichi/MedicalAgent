from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from backend.services.dep001a_semantic_safety_training import train_and_evaluate


if __name__ == "__main__":
    result = train_and_evaluate()
    print(json.dumps({
        "status": result["status"],
        "architecture": result["architecture"],
        "development": result["development"],
        "validation": result["validation"],
    }, indent=2))
