from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.dep001b_runtime_assurance import build_dep001b_runtime_assurance
from backend.services.dep001c_candidate_snapshot import mint_dep001c_candidate
from backend.services.dep001c_integrity_assurance import run_integrity_fault_injection


if __name__ == "__main__":
    preflight = ROOT / "Data/evals/safety/dep001c/preflight"
    runtime_path = preflight / "runtime_assurance.json"
    integrity_path = preflight / "integrity_fault_injection.json"
    runtime = build_dep001b_runtime_assurance(runtime_path)
    integrity = run_integrity_fault_injection(integrity_path)
    manifest = mint_dep001c_candidate(
        runtime_assurance_path=runtime_path,
        integrity_fault_injection_path=integrity_path,
    )
    print(json.dumps({
        "candidate_id": manifest["candidate_id"],
        "frozen_artifact_count": manifest["frozen_artifact_count"],
        "runtime_assurance_status": runtime["status"],
        "integrity_fault_injection_status": integrity["status"],
    }, indent=2))
