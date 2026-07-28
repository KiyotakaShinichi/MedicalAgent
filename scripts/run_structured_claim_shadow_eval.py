from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.structured_claim_shadow_eval import build_structured_claim_shadow_eval  # noqa: E402


if __name__ == "__main__":
    report = build_structured_claim_shadow_eval()
    print(json.dumps({"status": report["status"], "n_cases": report["n_cases"], "passed_n": report["passed_n"]}, indent=2))
