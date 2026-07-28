from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.synthetic_causal_v3_stress import build_synthetic_causal_v3_stress


if __name__ == "__main__":
    report = build_synthetic_causal_v3_stress()
    print(json.dumps({"status": report["status"], "seed_count": report["seed_count"], "decision": report["model_promotion_decision"]}, indent=2))
