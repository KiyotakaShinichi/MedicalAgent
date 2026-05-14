import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.evaluation_narrative_report import build_ai_ml_narrative_report  # noqa: E402


def main():
    report = build_ai_ml_narrative_report()
    print(json.dumps({
        "status": "generated",
        "files": report.get("files"),
        "summary": report.get("executive_summary"),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
