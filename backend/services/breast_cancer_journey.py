"""Breast-cancer journey phase model for organizing timeline events."""

from __future__ import annotations

from typing import Any, Mapping


JOURNEY_PHASE_VERSION = "breast_cancer_journey_phase_v1_2026_05"


JOURNEY_PHASES: dict[str, dict[str, Any]] = {
    "diagnosis_baseline": {
        "order": 1,
        "signals": ("diagnosis", "baseline imaging", "biopsy", "pathology"),
        "allowed_ai_role": "organize baseline context and missing data.",
    },
    "biomarker_pathology_workup": {
        "order": 2,
        "signals": ("ER", "PR", "HER2", "Ki-67", "IHC", "FISH"),
        "allowed_ai_role": "extract stated biomarkers and route for review.",
    },
    "genetic_counseling_readiness": {
        "order": 3,
        "signals": ("BRCA", "germline", "family history", "VUS"),
        "allowed_ai_role": "organize hereditary-risk context; do not interpret genetic risk.",
    },
    "treatment_cycles": {
        "order": 4,
        "signals": ("cycle", "chemotherapy", "paclitaxel", "doxorubicin", "trastuzumab"),
        "allowed_ai_role": "track cycle timing and side-effect context.",
    },
    "monitoring_side_effects": {
        "order": 5,
        "signals": ("CBC", "WBC", "ANC", "hemoglobin", "platelets", "symptom"),
        "allowed_ai_role": "surface monitoring signals and urgent review flags.",
    },
    "imaging_response_review": {
        "order": 6,
        "signals": ("MRI", "CT", "ultrasound", "impression", "lesion", "ascites"),
        "allowed_ai_role": "summarize report wording; do not diagnose response.",
    },
    "survivorship_supportive_care": {
        "order": 7,
        "signals": ("fatigue", "neuropathy", "lymphedema", "sleep", "nutrition", "exercise"),
        "allowed_ai_role": "support question preparation and symptom organization.",
    },
    "recurrence_metastatic_monitoring": {
        "order": 8,
        "signals": ("tumor marker", "CA 15-3", "CA 27.29", "CEA", "metastatic"),
        "allowed_ai_role": "trend/context only; never confirm recurrence.",
    },
}


def infer_journey_phase(event: Mapping[str, Any] | str) -> dict[str, Any]:
    text = event if isinstance(event, str) else " ".join(str(v) for v in event.values() if v is not None)
    lowered = text.lower()
    matches: list[tuple[str, int]] = []
    for phase, spec in JOURNEY_PHASES.items():
        score = sum(1 for signal in spec["signals"] if str(signal).lower() in lowered)
        if score:
            matches.append((phase, score))
    if not matches:
        phase = "monitoring_side_effects"
        confidence = "low"
    else:
        matches.sort(key=lambda item: (item[1], -JOURNEY_PHASES[item[0]]["order"]), reverse=True)
        phase = matches[0][0]
        confidence = "high" if matches[0][1] >= 2 else "moderate"
    spec = JOURNEY_PHASES[phase]
    return {
        "phase": phase,
        "confidence": confidence,
        "order": spec["order"],
        "allowed_ai_role": spec["allowed_ai_role"],
        "journey_phase_version": JOURNEY_PHASE_VERSION,
        "claim_boundary": "Journey phase organizes workflow only; it does not diagnose or stage cancer.",
    }


def journey_phase_manifest() -> dict[str, Any]:
    return {
        "version": JOURNEY_PHASE_VERSION,
        "phases": JOURNEY_PHASES,
        "claim_boundary": "Workflow organization only, not diagnosis or treatment planning.",
    }


__all__ = [
    "JOURNEY_PHASES",
    "JOURNEY_PHASE_VERSION",
    "infer_journey_phase",
    "journey_phase_manifest",
]
