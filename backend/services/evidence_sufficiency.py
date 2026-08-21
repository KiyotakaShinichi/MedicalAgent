"""Evidence-sufficiency layer for the synthetic monitoring model.

What this is
------------
The trained classifier always emits a probability between 0 and 1.  That number
is **only meaningful when the input row actually carries enough signal** to
support a response/toxicity claim.  In real monitoring data, some patients
will be missing imaging, others will be missing nadir labs, others will only
have pre-cycle CBC.  Forcing a confident prediction on those rows is exactly
the kind of silent-failure pattern this project is trying to avoid.

This module sits between feature ingestion and the classifier head and answers
two questions:

  1. Which feature *modalities* are actually present on this row?
  2. Given the question being asked (response classification, toxicity risk,
     urgent intervention), is the available evidence sufficient?

If the answer to (2) is "no", the caller should abstain — return
``decision="insufficient_evidence"`` with a structured reason and refuse to
quote a calibrated probability for the patient.

This is engineering scaffolding, not clinical decision support.  The rules
encoded here are explicit defaults a clinical advisor can override; nothing
here should be quoted as a clinical recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import math


# ─── Modality groups ──────────────────────────────────────────────────────────
# Mirror the feature lists in complete_synthetic_training.NUMERIC_FEATURES /
# CATEGORICAL_FEATURES so a future schema change cannot silently desync the
# two sides.  These groups are how the abstention layer reasons about "what
# kind of evidence is in this row" — they are NOT separate models.

MODALITY_GROUPS: dict[str, tuple[str, ...]] = {
    "demographics": ("age", "cycle", "stage", "molecular_subtype", "regimen"),
    "cbc_pre":      ("pre_wbc", "pre_anc", "pre_hemoglobin", "pre_platelets"),
    "cbc_nadir":    ("nadir_wbc", "nadir_anc", "nadir_hemoglobin", "nadir_platelets"),
    "cbc_recovery": ("recovery_wbc", "recovery_hemoglobin", "recovery_platelets"),
    "imaging":      ("mri_tumor_size_cm", "mri_percent_change_from_baseline"),
    "symptoms":     ("max_symptom_severity", "symptom_count"),
    "interventions": ("intervention_count", "dose_delayed", "dose_reduced"),
}

# A modality counts as "present" when at least N of its constituent fields
# carry a finite numeric value.  We require ≥1 by default so a row with even
# a single signal in the group still counts, but the threshold can be tuned
# per group if a clinical advisor wants stricter rules.
DEFAULT_PRESENCE_THRESHOLD: dict[str, int] = {
    "demographics":  2,  # need age + cycle at minimum to even position the row
    "cbc_pre":       1,
    "cbc_nadir":     1,
    "cbc_recovery":  1,
    "imaging":       1,
    "symptoms":      1,
    "interventions": 1,
}


# ─── Sufficiency rules per supported question ────────────────────────────────
# Rules are deliberately strict: response classification is the highest-risk
# model head, so patient-facing response-pattern and response-score outputs
# require imaging evidence. A complete longitudinal CBC can improve confidence
# when imaging is present, but CBC-only rows should route to lab/toxicity/review
# signals instead of forcing a response-pattern estimate. Toxicity and
# urgent-intervention signals are downstream of cycle-level monitoring and can
# be answered from CBC + symptoms alone.

SufficiencyDecision = str  # "sufficient" | "partial" | "insufficient"


@dataclass
class EvidenceAssessment:
    modalities_present: list[str]
    modalities_missing: list[str]
    sufficiency: SufficiencyDecision  # "sufficient" / "partial" / "insufficient"
    abstain: bool
    reason: str | None
    confidence_modifier: float  # multiplier to apply to model probability spread

    def to_dict(self) -> dict[str, object]:
        return {
            "modalities_present": list(self.modalities_present),
            "modalities_missing": list(self.modalities_missing),
            "sufficiency": self.sufficiency,
            "abstain": self.abstain,
            "reason": self.reason,
            "confidence_modifier": self.confidence_modifier,
        }


def _is_present(value: object, *, numeric: bool) -> bool:
    """A field counts as present when it carries usable signal.

    Rules depend on whether the field is declared numeric or categorical:
      - numeric fields require a finite float (NaN / None / unparseable
        strings like "n/a" count as missing),
      - categorical fields require a non-empty trimmed string.
    """
    if value is None:
        return False
    if numeric:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return False
            try:
                value = float(stripped)
            except ValueError:
                # Unparseable strings in numeric fields are missing data.
                return False
        if isinstance(value, (int, float)):
            return math.isfinite(float(value))
        # Anything else in a numeric slot (object, list, etc.) is treated
        # as missing rather than silently passing.
        return False
    # Categorical fields: any non-empty string is present.
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


# Lazy import to avoid circular dependency with complete_synthetic_training
# at module-load time — that module imports a fair amount of sklearn etc.
def _numeric_field_set() -> frozenset[str]:
    from backend.services.complete_synthetic_training import NUMERIC_FEATURES
    return frozenset(NUMERIC_FEATURES)


def _count_present(row: Mapping[str, object], fields: Iterable[str]) -> int:
    numeric = _numeric_field_set()
    return sum(
        1 for f in fields
        if _is_present(row.get(f), numeric=(f in numeric))
    )


def detect_modalities(
    row: Mapping[str, object],
    presence_threshold: Mapping[str, int] | None = None,
) -> tuple[list[str], list[str]]:
    """Return ``(present_modalities, missing_modalities)`` for a single row."""
    thresholds = {**DEFAULT_PRESENCE_THRESHOLD, **(presence_threshold or {})}
    present: list[str] = []
    missing: list[str] = []
    for group, fields in MODALITY_GROUPS.items():
        count = _count_present(row, fields)
        if count >= thresholds.get(group, 1):
            present.append(group)
        else:
            missing.append(group)
    return present, missing


# Rule definitions per supported question.  Each rule reports
# (sufficiency_decision, abstain_flag, reason_when_insufficient, confidence_modifier).
# `confidence_modifier` is a [0..1] multiplier the caller can apply to the
# model's probability *spread* around 0.5 when sufficiency is "partial" —
# moving the prediction closer to the prior reduces overconfidence on weak
# evidence without forcing a hard abstention.


def assess_response_classification(
    present: list[str],
    missing: list[str],
) -> EvidenceAssessment:
    """Response classification is the strongest claim the system makes — it
    requires imaging; complete longitudinal CBC is confidence context."""
    has_imaging = "imaging" in present
    longitudinal_cbc = all(
        group in present for group in ("cbc_pre", "cbc_nadir", "cbc_recovery")
    )
    has_demographics = "demographics" in present

    if not has_demographics:
        return EvidenceAssessment(
            modalities_present=present,
            modalities_missing=missing,
            sufficiency="insufficient",
            abstain=True,
            reason="missing_minimum_context",
            confidence_modifier=0.0,
        )

    if not has_imaging:
        return EvidenceAssessment(
            modalities_present=present,
            modalities_missing=missing,
            sufficiency="insufficient",
            abstain=True,
            reason="response_imaging_required_for_response_pattern",
            confidence_modifier=0.0,
        )

    # Both response signals present → highest confidence regime.
    if has_imaging and longitudinal_cbc:
        return EvidenceAssessment(
            modalities_present=present,
            modalities_missing=missing,
            sufficiency="sufficient",
            abstain=False,
            reason=None,
            confidence_modifier=1.0,
        )

    # Exactly one of the two response signals is present → partial evidence.
    return EvidenceAssessment(
        modalities_present=present,
        modalities_missing=missing,
        sufficiency="partial",
        abstain=False,
        reason="response_imaging_only_without_longitudinal_cbc",
        confidence_modifier=0.6,
    )


def assess_response_regression(
    present: list[str],
    missing: list[str],
) -> EvidenceAssessment:
    """Response-score regression uses the same minimum evidence contract as
    response classification, but it is intentionally named separately so
    traces and tests can distinguish the continuous score from the binary
    response-pattern head.

    Keeping this as an explicit assessor prevents the regression head from
    silently inheriting future classification-specific behavior that may not
    be appropriate for a 0-1 response-strength estimate.
    """
    return assess_response_classification(present, missing)


def assess_toxicity_classification(
    present: list[str],
    missing: list[str],
) -> EvidenceAssessment:
    """Toxicity risk needs CBC of any kind or symptom signal — cycle-level
    monitoring data is sufficient even when imaging is not available."""
    has_demographics = "demographics" in present
    has_any_cbc = any(group in present for group in ("cbc_pre", "cbc_nadir", "cbc_recovery"))
    has_symptoms = "symptoms" in present

    if not has_demographics:
        return EvidenceAssessment(
            modalities_present=present,
            modalities_missing=missing,
            sufficiency="insufficient",
            abstain=True,
            reason="missing_minimum_context",
            confidence_modifier=0.0,
        )

    if not has_any_cbc and not has_symptoms:
        return EvidenceAssessment(
            modalities_present=present,
            modalities_missing=missing,
            sufficiency="insufficient",
            abstain=True,
            reason="no_toxicity_signal_cbc_or_symptoms_required",
            confidence_modifier=0.0,
        )

    if has_any_cbc and has_symptoms:
        return EvidenceAssessment(
            modalities_present=present,
            modalities_missing=missing,
            sufficiency="sufficient",
            abstain=False,
            reason=None,
            confidence_modifier=1.0,
        )

    return EvidenceAssessment(
        modalities_present=present,
        modalities_missing=missing,
        sufficiency="partial",
        abstain=False,
        reason="toxicity_signal_from_single_modality_only",
        confidence_modifier=0.75,
    )


def assess_urgent_intervention(
    present: list[str],
    missing: list[str],
) -> EvidenceAssessment:
    """Urgent intervention signals are routed straight to clinician review
    whenever symptoms or critical CBC values are present, but require at
    least one of those to be present at all."""
    has_demographics = "demographics" in present
    has_acute_signal = ("symptoms" in present) or ("cbc_nadir" in present)
    if not has_demographics or not has_acute_signal:
        return EvidenceAssessment(
            modalities_present=present,
            modalities_missing=missing,
            sufficiency="insufficient",
            abstain=True,
            reason="no_acute_signal_symptoms_or_nadir_cbc_required",
            confidence_modifier=0.0,
        )
    if "symptoms" in present and "cbc_nadir" in present:
        modifier = 1.0
        sufficiency = "sufficient"
    else:
        modifier = 0.7
        sufficiency = "partial"
    return EvidenceAssessment(
        modalities_present=present,
        modalities_missing=missing,
        sufficiency=sufficiency,
        abstain=False,
        reason=None if sufficiency == "sufficient" else "urgent_signal_from_single_modality_only",
        confidence_modifier=modifier,
    )


QUESTION_ASSESSORS = {
    "response_classification": assess_response_classification,
    "response_regression":     assess_response_regression,
    "toxicity_classification": assess_toxicity_classification,
    "urgent_intervention":     assess_urgent_intervention,
}


def assess_evidence(
    row: Mapping[str, object],
    *,
    question: str = "response_classification",
    presence_threshold: Mapping[str, int] | None = None,
) -> EvidenceAssessment:
    """Single-row helper: detect modalities then apply the question-specific
    sufficiency rule.  Defaults to the strictest question (response)."""
    if question not in QUESTION_ASSESSORS:
        raise ValueError(
            f"Unknown question '{question}'. Supported: {sorted(QUESTION_ASSESSORS)}",
        )
    present, missing = detect_modalities(row, presence_threshold=presence_threshold)
    return QUESTION_ASSESSORS[question](present, missing)


__all__ = [
    "MODALITY_GROUPS",
    "EvidenceAssessment",
    "assess_evidence",
    "assess_response_classification",
    "assess_response_regression",
    "assess_toxicity_classification",
    "assess_urgent_intervention",
    "detect_modalities",
]
