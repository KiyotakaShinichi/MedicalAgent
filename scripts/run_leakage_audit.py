"""Run the unified leakage audit and write the artifact.

Exits with code 1 when any check fails so CI can gate on it.  Prints a small
JSON summary on stdout for human/CI consumption.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.leakage_audit import run_leakage_audit


if __name__ == "__main__":
    payload = run_leakage_audit()
    summary = {
        "output_path": "Data/evals/models/latest_leakage_audit.json",
        "status": payload.get("status"),
        "checks_passed": payload.get("summary", {}).get("checks_passed"),
        "checks_failed": payload.get("summary", {}).get("checks_failed"),
        "failed_checks": [
            item["name"] for item in payload.get("findings", []) if item["status"] != "passed"
        ],
        "temporal_sub_audit_status": payload.get("temporal_sub_audit", {}).get("status"),
    }
    print(json.dumps(summary, indent=2))
    sys.exit(0 if payload.get("status") == "passed" else 1)
