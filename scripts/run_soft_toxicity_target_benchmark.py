from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.services.soft_toxicity_target_benchmark import run_soft_toxicity_target_benchmark


if __name__ == "__main__":
    r = run_soft_toxicity_target_benchmark()
    print(json.dumps({"status": r["status"], "auroc": r["soft_target_model"]["auroc"]}, indent=2))
