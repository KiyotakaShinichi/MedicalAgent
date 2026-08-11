from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.human_review_feedback_ingestion import build_human_review_feedback_ingestion


if __name__ == "__main__":
    result = build_human_review_feedback_ingestion()
    print(json.dumps({
        "status": result["status"],
        "external_review_completed": result["external_review_completed"],
        "accepted_feedback_row_count": result["accepted_feedback_row_count"],
    }, indent=2))
