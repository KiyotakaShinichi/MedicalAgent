from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.dep001c_integrity_assurance import run_integrity_fault_injection
from backend.services.dep001d_candidate_snapshot import mint_dep001d_candidate
from backend.services.dep001d_development_assurance import build_dep001d_development_assurance
from backend.services.dep001d_fault_injection import run_dep001d_fault_injection


if __name__ == "__main__":
    base = ROOT / "Data/evals/safety/dep001d"
    fault_path = base / "latest_fault_injection.json"
    integrity_path = base / "latest_integrity_fault_injection.json"
    assurance_path = base / "latest_development_assurance.json"
    overlap_path = base / "latest_development_overlap_audit.json"
    fault = run_dep001d_fault_injection(fault_path)
    integrity = run_integrity_fault_injection(integrity_path)
    assurance = build_dep001d_development_assurance(assurance_path)
    manifest = mint_dep001d_candidate(
        development_assurance_path=assurance_path,
        fault_injection_path=fault_path,
        integrity_fault_injection_path=integrity_path,
        overlap_audit_path=overlap_path,
    )
    print(json.dumps({
        "candidate_id": manifest["candidate_id"],
        "frozen_artifact_count": manifest["frozen_artifact_count"],
        "development_assurance_status": assurance["status"],
        "fault_injection_status": fault["status"],
        "integrity_fault_injection_status": integrity["status"],
    }, indent=2))
