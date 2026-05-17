"""Minimum evidence standards for patient-support answers and model signals."""

from __future__ import annotations

from typing import Any


EVIDENCE_STANDARDS_VERSION = "minimum_evidence_standards_v1_2026_05"


MINIMUM_EVIDENCE_STANDARDS: dict[str, dict[str, Any]] = {
    "general_education": {
        "required": ("trusted_kb_source",),
        "answer_policy": "RAG answer with citation; no patient-specific decision.",
    },
    "response_classification": {
        "required_any": ("imaging", "complete_longitudinal_cbc"),
        "required_all": ("demographics",),
        "answer_policy": "May emit monitor-only pattern signal; abstain if missing response evidence.",
    },
    "response_regression": {
        "required_any": ("imaging", "complete_longitudinal_cbc"),
        "required_all": ("demographics",),
        "answer_policy": "May emit score only with uncertainty band and claim boundary.",
    },
    "toxicity_classification": {
        "required_any": ("any_cbc", "symptoms"),
        "required_all": ("demographics",),
        "answer_policy": "May emit toxicity monitoring signal; urgent symptoms route to care team.",
    },
    "genetic_result_organization": {
        "required_any": ("uploaded_report", "typed_report_fields"),
        "answer_policy": "Organize fields only; route to genetic counselor/clinician review.",
    },
    "biomarker_explanation": {
        "required_any": ("pathology_report_text", "typed_biomarker_fields", "trusted_kb_source"),
        "answer_policy": "Explain terminology only; no treatment recommendation.",
    },
    "tumor_marker_trend": {
        "required_any": ("marker_value", "trend_values"),
        "answer_policy": "Context-dependent monitoring only; never diagnose recurrence.",
    },
    "treatment_decision": {
        "required": (),
        "answer_policy": "Blocked. Refer to oncology team.",
    },
    "diagnosis_or_prognosis": {
        "required": (),
        "answer_policy": "Blocked. Refer to clinician/emergency services when urgent.",
    },
}


def standards_manifest() -> dict[str, Any]:
    return {
        "version": EVIDENCE_STANDARDS_VERSION,
        "standards": MINIMUM_EVIDENCE_STANDARDS,
        "claim_boundary": (
            "Minimum evidence standards are engineering guardrails that decide "
            "when the system may answer, must hedge, or must abstain. They are "
            "not clinical practice guidelines."
        ),
    }


def standard_for(question_type: str) -> dict[str, Any]:
    return {
        "question_type": question_type,
        **MINIMUM_EVIDENCE_STANDARDS.get(question_type, {
            "required": (),
            "answer_policy": "Unknown question type; route to clinician review or safe general help.",
        }),
        "standards_version": EVIDENCE_STANDARDS_VERSION,
    }


__all__ = [
    "EVIDENCE_STANDARDS_VERSION",
    "MINIMUM_EVIDENCE_STANDARDS",
    "standard_for",
    "standards_manifest",
]
