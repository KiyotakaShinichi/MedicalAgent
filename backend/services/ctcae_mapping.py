from __future__ import annotations

from typing import Any


URGENT_SYMPTOM_TERMS = {
    "fever",
    "chest pain",
    "shortness of breath",
    "difficulty breathing",
    "uncontrolled bleeding",
    "blood discharge",
    "severe headache",
    "confusion",
    "fainting",
}

CLAIM_BOUNDARY = (
    "This is a patient-reported severity review hint, not a clinician-assigned CTCAE grade. "
    "Formal adverse-event grading requires clinician review and clinical context."
)


def map_symptom_to_ctcae_review_hint(
    *,
    symptom: str,
    severity: int,
    urgent_flag: bool = False,
    notes: str | None = None,
) -> dict[str, Any]:
    """Map a 0-10 patient severity entry into a safe clinician-review hint."""

    symptom_text = f"{symptom or ''} {notes or ''}".lower()
    red_flag_terms = sorted(term for term in URGENT_SYMPTOM_TERMS if term in symptom_text)
    urgent_review = bool(urgent_flag or severity >= 8 or red_flag_terms)
    bucket = _patient_bucket(severity)
    ctcae_hint = _ctcae_hint(severity, urgent_review)
    return {
        "schema_version": "ctcae_review_hint_v1",
        "patient_severity": int(severity),
        "patient_severity_bucket": bucket,
        "ctcae_hint": ctcae_hint,
        "urgent_review": urgent_review,
        "red_flag_terms": red_flag_terms,
        "review_focus": _review_focus(bucket, urgent_review, red_flag_terms),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _patient_bucket(severity: int) -> str:
    if severity <= 3:
        return "mild_patient_reported"
    if severity <= 6:
        return "moderate_patient_reported"
    return "severe_patient_reported"


def _ctcae_hint(severity: int, urgent_review: bool) -> str:
    if urgent_review:
        return "grade_3_or_higher_review_hint"
    if severity <= 3:
        return "grade_1_review_hint"
    if severity <= 6:
        return "grade_2_review_hint"
    return "grade_3_review_hint"


def _review_focus(bucket: str, urgent_review: bool, red_flags: list[str]) -> list[str]:
    focus = ["Confirm onset, duration, treatment-cycle timing, and associated symptoms."]
    if urgent_review:
        focus.append("Review promptly with the oncology care team or local emergency pathway if clinically appropriate.")
    if red_flags:
        focus.append(f"Patient text contains red-flag term(s): {', '.join(red_flags)}.")
    if bucket == "severe_patient_reported":
        focus.append("Assess functional impact and need for clinician-assigned adverse-event grading.")
    return focus


__all__ = ["map_symptom_to_ctcae_review_hint", "CLAIM_BOUNDARY", "URGENT_SYMPTOM_TERMS"]
