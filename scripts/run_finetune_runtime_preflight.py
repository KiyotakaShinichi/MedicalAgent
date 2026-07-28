from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.finetune_runtime_preflight import build_finetune_runtime_preflight


if __name__ == "__main__":
    report = build_finetune_runtime_preflight()
    print(json.dumps({"status": report["status"], "model_trained": report["model_trained"], "runtime": report["runtime_probe"]["status"]}, indent=2))
