from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.xai_comprehension_contract_eval import build_xai_comprehension_contract_eval


if __name__ == "__main__":
    report = build_xai_comprehension_contract_eval()
    print(json.dumps({"status": report["status"], "pass_rate": report["pass_rate"], "human_study": report["human_participant_study_completed"]}, indent=2))
