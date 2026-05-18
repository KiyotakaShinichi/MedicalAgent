from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.clinical_ontology import ontology_manifest
from backend.services.ctcae_mapping import URGENT_SYMPTOM_TERMS
from backend.services.genetic_counseling import GENETIC_BOUNDARY_NOTE, GENETIC_UNSAFE_PHRASES
from backend.services.lab_reference_context import build_cbc_reference_context
from backend.services.medical_claim_boundary import claim_boundary_manifest
from backend.services.medical_evidence_standards import standards_manifest
from backend.services.medication_interactions import RULES as INTERACTION_RULES


DEFAULT_JSON_PATH = "Data/evals/medical/latest_medical_advisor_review_packet.json"
DEFAULT_MD_PATH = "docs/medical_advisor_review_packet.md"


def build_medical_advisor_review_packet(
    output_path: str = DEFAULT_JSON_PATH,
    md_path: str = DEFAULT_MD_PATH,
) -> dict[str, Any]:
    packet = {
        "schema_version": "medical_advisor_review_packet_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_clinical_advisor_review",
        "claim_boundary": (
            "This packet is for expert review of an engineering prototype. It is not clinical sign-off, "
            "clinical validation, or authorization for patient care use."
        ),
        "review_requested_for": [
            "urgent symptom/red-flag vocabulary",
            "patient-reported severity to CTCAE-style review hints",
            "CBC population-default reference context and limitations",
            "supplement/medication interaction review-routing rules",
            "genetics/VUS/tumor-marker refusal and education boundaries",
            "minimum evidence standards for model and RAG outputs",
        ],
        "clinical_ontology": ontology_manifest(),
        "minimum_evidence_standards": standards_manifest(),
        "medical_claim_boundaries": claim_boundary_manifest(),
        "cbc_reference_context_example": build_cbc_reference_context(wbc=2.8, hemoglobin=11.4, platelets=120),
        "urgent_symptom_terms": sorted(URGENT_SYMPTOM_TERMS),
        "interaction_rule_count": len(INTERACTION_RULES),
        "interaction_rules": [
            {
                "rule_id": rule.id,
                "trigger_terms": list(rule.trigger_terms),
                "context_terms": list(rule.context_terms),
                "severity": rule.severity,
                "message": rule.message,
                "clinician_action": rule.clinician_action,
            }
            for rule in INTERACTION_RULES
        ],
        "genetics": {
            "boundary_note": GENETIC_BOUNDARY_NOTE,
            "unsafe_phrases": GENETIC_UNSAFE_PHRASES,
            "expanded_family_history_fields": [
                "bilateral_breast_cancer",
                "multiple_primary_cancers",
                "ancestry_ethnicity",
                "prior_breast_biopsy_atypia",
                "relation_degree",
            ],
        },
        "advisor_questions": [
            "Which urgent symptom terms are missing or too broad?",
            "Are CBC reference-context warnings phrased safely for patients?",
            "Which supplement interaction rules should be added, removed, or clinician-only?",
            "Are genetics/VUS/tumor-marker boundaries conservative enough?",
            "Which family-history fields are required before genetic-counseling readiness is credible?",
        ],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(packet, indent=2), encoding="utf-8")
    Path(md_path).parent.mkdir(parents=True, exist_ok=True)
    Path(md_path).write_text(_markdown(packet), encoding="utf-8")
    return packet


def _markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# Medical Advisor Review Packet",
        "",
        f"Generated: {packet['generated_at']}",
        "",
        "## Claim Boundary",
        packet["claim_boundary"],
        "",
        "## Review Scope",
    ]
    lines.extend(f"- {item}" for item in packet["review_requested_for"])
    lines.extend([
        "",
        "## Urgent Symptom Terms",
        ", ".join(packet["urgent_symptom_terms"]),
        "",
        "## Genetics Boundary",
        packet["genetics"]["boundary_note"],
        "",
        "## Expanded Family-History Fields",
    ])
    lines.extend(f"- {field}" for field in packet["genetics"]["expanded_family_history_fields"])
    lines.extend([
        "",
        "## Supplement/Medication Interaction Rules",
    ])
    lines.extend(f"- {rule['rule_id']}: {rule['message']}" for rule in packet["interaction_rules"])
    lines.extend([
        "",
        "## Questions For Advisor",
    ])
    lines.extend(f"- {q}" for q in packet["advisor_questions"])
    lines.append("")
    return "\n".join(lines)


__all__ = ["build_medical_advisor_review_packet", "DEFAULT_JSON_PATH", "DEFAULT_MD_PATH"]
