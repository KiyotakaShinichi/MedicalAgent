from __future__ import annotations

from typing import Any

from backend.services.ctcae_mapping import CLAIM_BOUNDARY, map_symptom_to_ctcae_review_hint


def map_toxicity_review_hint(
    *,
    symptom: str | None = None,
    severity: int | None = None,
    anc: float | None = None,
    platelets: float | None = None,
    vomiting_count: int | None = None,
    bleeding: bool = False,
    neuropathy_functional_impact: bool = False,
    notes: str | None = None,
) -> dict[str, Any]:
    symptom_text = symptom or ("bleeding" if bleeding else "lab_abnormality")
    severity_value = int(severity if severity is not None else 0)
    base = map_symptom_to_ctcae_review_hint(symptom=symptom_text, severity=severity_value, urgent_flag=bleeding, notes=notes)
    lab_flags: list[str] = []
    if anc is not None and anc < 1.0:
        lab_flags.append("low_anc_review")
    if platelets is not None and platelets < 50:
        lab_flags.append("low_platelets_review")
    if vomiting_count is not None and vomiting_count >= 3:
        lab_flags.append("repeated_vomiting_review")
    if neuropathy_functional_impact:
        lab_flags.append("neuropathy_functional_impact_review")
    priority = "urgent_review" if base["urgent_review"] or bleeding or "low_anc_review" in lab_flags else "routine_review"
    if lab_flags and priority != "urgent_review":
        priority = "elevated_review"
    return {
        **base,
        "schema_version": "toxicity_review_hint_v1",
        "lab_or_context_flags": lab_flags,
        "review_priority": priority,
        "safe_label": "Review severity hint",
        "patient_safe_phrase": (
            "This pattern resembles a higher-severity review category and should be reviewed by a clinician."
            if priority != "routine_review"
            else "This is an organizing hint for clinician review, not a toxicity diagnosis."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


__all__ = ["map_toxicity_review_hint"]
