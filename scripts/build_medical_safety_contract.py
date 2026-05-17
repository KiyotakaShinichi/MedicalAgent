"""Generate the medical safety contract artifact."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.medical_safety_contract import build_medical_safety_contract


if __name__ == "__main__":
    payload = build_medical_safety_contract()
    print(json.dumps({
        "status": payload.get("status"),
        "ontology_version": payload["clinical_ontology"]["version"],
        "evidence_standards_version": payload["minimum_evidence_standards"]["version"],
        "claim_boundary_version": payload["medical_claim_boundary"]["version"],
        "journey_phase_version": payload["journey_phase_model"]["version"],
    }, indent=2))
    sys.exit(0 if payload.get("status") == "strong" else 1)
