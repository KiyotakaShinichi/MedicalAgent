import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.minimum_evidence import build_minimum_evidence_standards_artifact


if __name__ == "__main__":
    payload = build_minimum_evidence_standards_artifact()
    print(json.dumps({"status": payload["status"], "standard_count": len(payload["standards"])}, indent=2))
    sys.exit(0 if payload["status"] == "strong" else 1)
