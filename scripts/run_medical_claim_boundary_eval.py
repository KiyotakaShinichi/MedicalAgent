import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.medical_claim_boundary_eval import run_medical_claim_boundary_eval


if __name__ == "__main__":
    payload = run_medical_claim_boundary_eval()
    print(json.dumps({"status": payload["status"], "summary": payload["summary"]}, indent=2))
    sys.exit(0 if payload["status"] == "strong" else 1)
