from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.clinical_safety_checklist import build_clinical_safety_review_checklist


if __name__ == "__main__":
    report = build_clinical_safety_review_checklist()
    print(json.dumps({
        "status": report.get("status"),
        "section_count": len(report.get("sections", [])),
    }, indent=2))
