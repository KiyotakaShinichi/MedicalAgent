from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.event_taxonomy import write_event_taxonomy_doc  # noqa: E402
from backend.services.governance_readiness_artifacts import (  # noqa: E402
    write_clinical_performance_dossier_status,
    write_medical_governance_artifacts,
    write_near_boundary_safety_eval,
    write_rag_gold_claim_grounding_cases,
    write_real_data_readiness_checklist,
    write_taglish_safety_goldset,
    write_uncertainty_dossier,
)
from backend.services.ops_health_snapshot import build_service_health_snapshot  # noqa: E402
from backend.services.semantic_citation_verifier import run_semantic_citation_verification_eval  # noqa: E402


def main() -> int:
    artifacts = {
        "rag_gold": write_rag_gold_claim_grounding_cases(),
        "semantic_citation": run_semantic_citation_verification_eval(),
        "taglish_goldset": write_taglish_safety_goldset(),
        "near_boundary": write_near_boundary_safety_eval(),
        "uncertainty": write_uncertainty_dossier(),
        "real_data_readiness": write_real_data_readiness_checklist(),
        "clinical_dossier": write_clinical_performance_dossier_status(),
        "event_taxonomy": write_event_taxonomy_doc(),
        "medical_governance": write_medical_governance_artifacts(),
        "ops_health": build_service_health_snapshot(),
    }
    print(json.dumps({
        "status": "strong",
        "artifact_count": len(artifacts),
        "created": sorted(artifacts),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
