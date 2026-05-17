from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.database import SessionLocal
from backend.services.system_health import build_system_health_report


if __name__ == "__main__":
    db = SessionLocal()
    try:
        report = build_system_health_report(db=db)
    finally:
        db.close()
    print(json.dumps({
        "status": report.get("status"),
        "issue_count": len(report.get("issues", [])),
    }, indent=2))
