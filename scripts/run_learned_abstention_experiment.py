from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.services.learned_abstention_experiment import run_learned_abstention_experiment


if __name__ == "__main__":
    r = run_learned_abstention_experiment()
    print(json.dumps({"status": r["status"], "auroc": r["abstention_head"]["auroc"]}, indent=2))
