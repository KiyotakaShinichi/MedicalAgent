from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from backend.services.artifact_manifest import build_artifact_manifest
from backend.services.medical_evidence_standards import MINIMUM_EVIDENCE_STANDARDS, EVIDENCE_STANDARDS_VERSION


DEFAULT_OUTPUT_PATH = "Data/evals/medical/latest_minimum_evidence_standards.json"


QUESTION_TYPE_ALIASES = {
    "response_pattern": "response_classification",
    "response_pattern_estimation": "response_classification",
    "toxicity_signal": "toxicity_classification",
    "urgent_symptom_escalation": "urgent_symptom_escalation",
    "genetic_counseling_readiness": "genetic_counseling_readiness",
    "supplement_interaction_safety": "supplement_interaction_safety",
    "imaging_report_explanation": "imaging_report_explanation",
}

EXTENDED_STANDARDS: dict[str, dict[str, Any]] = {
    **MINIMUM_EVIDENCE_STANDARDS,
    "urgent_symptom_escalation": {
        "required_any": ("urgent_symptom_text", "fever", "bleeding", "shortness_of_breath", "chest_pain"),
        "optional": ("recent_cbc", "treatment_cycle_timing"),
        "insufficiency_behavior": "Escalate based on red-flag wording; do not wait for complete data.",
        "review_routing": "oncology_team_or_emergency_pathway",
        "blocked_conclusions": ("diagnosis", "home_care_only_reassurance", "dose_change"),
    },
    "genetic_counseling_readiness": {
        "required_any": ("family_history", "genetic_test_record", "known_familial_mutation", "patient_question"),
        "optional": ("pathology_biomarkers", "age_at_diagnosis", "maternal_paternal_side"),
        "insufficiency_behavior": "Organize known fields and list missing fields; route to genetics-trained clinician/genetic counselor.",
        "review_routing": "genetic_counselor_or_oncology_team",
        "blocked_conclusions": ("genetic_diagnosis", "relative_risk_prediction", "treatment_change"),
    },
    "supplement_interaction_safety": {
        "required_any": ("supplement_name", "medication_list", "patient_question"),
        "optional": ("current_regimen", "platelet_trend", "anticoagulant_use"),
        "insufficiency_behavior": "Flag for oncology/pharmacist review; never call supplement safe with treatment.",
        "review_routing": "oncology_pharmacist_or_care_team",
        "blocked_conclusions": ("safe_to_take", "replace_treatment", "dose_instruction"),
    },
    "imaging_report_explanation": {
        "required_any": ("report_text", "modality", "impression"),
        "optional": ("prior_imaging", "treatment_cycle_timing", "pathology_context"),
        "insufficiency_behavior": "Explain terms only and ask care team to interpret patient-specific implications.",
        "review_routing": "oncology_team_or_radiology_report_review",
        "blocked_conclusions": ("diagnosis", "progression_confirmation", "treatment_response_confirmation"),
    },
}


def evaluate_minimum_evidence(question_type: str, modalities_present: Iterable[str]) -> dict[str, Any]:
    key = QUESTION_TYPE_ALIASES.get(question_type, question_type)
    standard = EXTENDED_STANDARDS.get(key)
    present = set(modalities_present or ())
    if not standard:
        return _decision(question_type, present, "unknown_question_type", False, "Route to clinician review or safe general help.")
    required_all = set(standard.get("required_all") or standard.get("required") or ())
    required_any = set(standard.get("required_any") or ())
    all_ok = required_all.issubset(present)
    any_ok = True if not required_any else bool(required_any & present)
    sufficient = all_ok and any_ok
    missing_all = sorted(required_all - present)
    missing_any = sorted(required_any) if required_any and not any_ok else []
    return {
        "question_type": key,
        "decision": "sufficient" if sufficient else "insufficient",
        "modalities_present": sorted(present),
        "missing_required_all": missing_all,
        "missing_required_any_options": missing_any,
        "insufficiency_behavior": standard.get("insufficiency_behavior") or standard.get("answer_policy"),
        "review_routing": standard.get("review_routing", "clinician_review"),
        "blocked_conclusions": list(standard.get("blocked_conclusions", ())),
        "standards_version": EVIDENCE_STANDARDS_VERSION,
        "claim_boundary": "Engineering minimum-evidence rule, not a clinical guideline.",
    }


def build_minimum_evidence_standards_artifact(output_path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    payload = {
        **build_artifact_manifest(dataset_paths={"standards": "backend/services/medical_evidence_standards.py"}),
        "schema_version": "minimum_evidence_standards_artifact_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong",
        "version": EVIDENCE_STANDARDS_VERSION,
        "standards": EXTENDED_STANDARDS,
        "claim_boundary": "Minimum evidence standards route answers; they do not validate clinical truth.",
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _decision(question_type: str, present: set[str], decision: str, sufficient: bool, behavior: str) -> dict[str, Any]:
    return {
        "question_type": question_type,
        "decision": "sufficient" if sufficient else "insufficient",
        "modalities_present": sorted(present),
        "reason": decision,
        "insufficiency_behavior": behavior,
        "standards_version": EVIDENCE_STANDARDS_VERSION,
    }


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "EXTENDED_STANDARDS",
    "build_minimum_evidence_standards_artifact",
    "evaluate_minimum_evidence",
]
