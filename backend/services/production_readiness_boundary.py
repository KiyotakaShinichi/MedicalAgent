"""Production-shaped readiness boundary.

This artifact intentionally refuses to call the project production-ready for
healthcare use. It checks engineering readiness signals and names the blockers
that require external review, real data, governance, and compliance.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_PATH = Path("Data/evals/governance/latest_production_readiness_boundary.json")
RELEASE_GATE = Path("Data/evals/governance/latest_release_gate_explanation.json")
EXTERNAL_REVIEW = Path("Data/evals/governance/latest_external_review_readiness.json")
REAL_DATA = Path("Data/evals/models/latest_real_data_readiness_checklist.json")

CLAIM_BOUNDARY = (
    "Production-shaped readiness is an engineering discipline artifact. NLCare "
    "is not healthcare-production-ready, not clinically validated, not approved "
    "for real patient care, and not compliant for PHI workflows."
)


def build_production_readiness_boundary(output_path: str | Path = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    release_gate = _read(RELEASE_GATE)
    external_review = _read(EXTERNAL_REVIEW)
    real_data = _read(REAL_DATA)
    checks = _checks(release_gate, external_review, real_data)
    blockers = [check for check in checks if check["blocks_healthcare_production"]]
    payload = {
        "schema_version": "production_readiness_boundary_v1_2026_05",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "production_shaped_not_healthcare_production_ready",
        "headline_metric": f"{len(blockers)} healthcare-production blockers remain",
        "total_n": len(checks),
        "pass_count": sum(1 for check in checks if check["engineering_status"] in {"present", "passing", "ready"}),
        "fail_count": len(blockers),
        "skipped_count": 0,
        "engineering_checks": checks,
        "healthcare_production_ready": False,
        "software_production_ready": False,
        "clinical_validation": False,
        "external_review_completed": bool(external_review.get("completed_external_review_count", 0)),
        "allowed_claim": (
            "Production-shaped engineering prototype with release gates, eval "
            "artifacts, traceability, and explicit clinical boundaries."
        ),
        "blocked_claims": [
            "production healthcare system",
            "clinically validated AI",
            "safe for real patient care",
            "clinician-approved",
            "FHIR/EHR interoperable product",
            "HIPAA/PHI compliant deployment",
            "treatment recommendation system",
        ],
        "required_before_healthcare_production": [
            "external-author adversarial/RAG eval completed",
            "clinician/nurse safety wording review",
            "genetic counselor review for VUS/genetics behavior",
            "senior MLE review of evaluation design",
            "real or public external cohort validation with exact-label mapping",
            "IRB/ethics governance for any real patient data",
            "formal PHI/privacy/security/compliance review",
            "deployment runbook with monitoring, rollback, incident response, and access controls",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _checks(release_gate: dict[str, Any], external_review: dict[str, Any], real_data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": "release_gate",
            "engineering_status": "passing" if release_gate.get("status") == "strong" else "needs_attention",
            "blocks_healthcare_production": False,
            "evidence": "Data/evals/governance/latest_release_gate_explanation.json",
        },
        {
            "name": "external_author_eval",
            "engineering_status": "ready" if external_review.get("status") == "ready_for_external_authoring" else "missing",
            "blocks_healthcare_production": True,
            "evidence": "External-author eval prepared but not completed.",
        },
        {
            "name": "clinician_review",
            "engineering_status": "not_completed",
            "blocks_healthcare_production": True,
            "evidence": "No clinician/nurse review log exists.",
        },
        {
            "name": "genetic_counselor_review",
            "engineering_status": "not_completed",
            "blocks_healthcare_production": True,
            "evidence": "No genetic-counselor VUS/genetics review completed.",
        },
        {
            "name": "real_data_readiness",
            "engineering_status": real_data.get("status", "not_ready"),
            "blocks_healthcare_production": True,
            "evidence": "No real patient cohort or clinician-reviewed labels.",
        },
        {
            "name": "ethics_irb",
            "engineering_status": "not_completed",
            "blocks_healthcare_production": True,
            "evidence": "No IRB/ethics approval.",
        },
        {
            "name": "phi_compliance",
            "engineering_status": "not_completed",
            "blocks_healthcare_production": True,
            "evidence": "No formal HIPAA/PHI/compliance review.",
        },
        {
            "name": "production_sre",
            "engineering_status": "partial",
            "blocks_healthcare_production": True,
            "evidence": "Local ops/latency/release artifacts exist; no production SLO or hosted clinical monitoring.",
        },
    ]


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


__all__ = ["build_production_readiness_boundary"]
