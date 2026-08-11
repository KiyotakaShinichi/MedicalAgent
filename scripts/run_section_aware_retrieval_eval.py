from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.section_aware_retrieval_eval import run_section_aware_retrieval_eval  # noqa: E402


if __name__ == "__main__":
    report = run_section_aware_retrieval_eval()
    print(json.dumps({
        "status": report["status"],
        "known_section_miss_count": report["known_section_miss_count"],
        "known_miss_evaluation": {
            key: value for key, value in report["known_miss_evaluation"].items() if key != "cases"
        },
        "decision": report["decision"],
    }, indent=2))
