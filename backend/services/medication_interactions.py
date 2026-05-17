from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


INTERACTION_CHECKER_VERSION = "medication_interaction_checker_v1_2026_05"


@dataclass(frozen=True)
class InteractionRule:
    id: str
    trigger_terms: tuple[str, ...]
    context_terms: tuple[str, ...]
    severity: str
    message: str
    clinician_action: str


RULES: tuple[InteractionRule, ...] = (
    InteractionRule(
        id="st_johns_wort_cyp_interaction",
        trigger_terms=("st johns wort", "st. johns wort", "st john's wort", "hypericum"),
        context_terms=("tamoxifen", "palbociclib", "ribociclib", "abemaciclib", "capecitabine", "docetaxel", "paclitaxel"),
        severity="pharmacist_review",
        message=(
            "St. John's wort can affect drug-metabolism pathways and may interact with oncology medicines. "
            "Do not start or stop it without oncology-team/pharmacist review."
        ),
        clinician_action="Review supplement use against the active medication list.",
    ),
    InteractionRule(
        id="grapefruit_cdk46_interaction",
        trigger_terms=("grapefruit", "grapefruit juice"),
        context_terms=("palbociclib", "ribociclib", "abemaciclib"),
        severity="pharmacist_review",
        message=(
            "Grapefruit products can interact with several oral cancer medicines. Ask the oncology team or pharmacist "
            "before using them with CDK4/6 inhibitors."
        ),
        clinician_action="Check whether the patient is taking a CYP3A-sensitive oral anticancer medicine.",
    ),
    InteractionRule(
        id="bleeding_risk_supplements",
        trigger_terms=("ginkgo", "garlic supplement", "high dose garlic", "turmeric", "curcumin"),
        context_terms=("warfarin", "apixaban", "rivaroxaban", "enoxaparin", "aspirin", "clopidogrel", "platelets"),
        severity="review_before_use",
        message=(
            "Some supplements may affect bleeding risk or procedures. This system cannot determine safety; "
            "review with the care team/pharmacist before use."
        ),
        clinician_action="Review bleeding risk, platelet trend, anticoagulants/antiplatelets, and procedure timing.",
    ),
    InteractionRule(
        id="cbd_cyp_interaction",
        trigger_terms=("cbd", "cannabidiol", "cannabis", "marijuana"),
        context_terms=("tamoxifen", "palbociclib", "ribociclib", "abemaciclib", "olaparib", "talazoparib", "anti nausea", "ondansetron"),
        severity="pharmacist_review",
        message=(
            "CBD/cannabis products may interact with medicines or increase sedation/nausea effects. "
            "Discuss use with the oncology team/pharmacist first."
        ),
        clinician_action="Check product, dose, legality/context, sedating medicines, and oral anticancer medicines.",
    ),
    InteractionRule(
        id="antioxidants_during_treatment",
        trigger_terms=("high dose vitamin c", "megadose vitamin c", "high dose antioxidant", "antioxidant supplement", "vitamin e supplement"),
        context_terms=("chemotherapy", "radiation", "doxorubicin", "cyclophosphamide", "paclitaxel", "docetaxel", "carboplatin"),
        severity="review_before_use",
        message=(
            "High-dose antioxidant supplements during chemotherapy or radiation should be reviewed with the oncology team. "
            "Do not use them as a replacement for prescribed treatment."
        ),
        clinician_action="Clarify supplement dose and timing relative to chemotherapy/radiation.",
    ),
)


def check_medication_interactions(
    new_medication: str,
    current_medications: Iterable[str] = (),
    *,
    notes: str | None = None,
) -> dict:
    """Return conservative supplement/drug interaction flags.

    This is not a medication-safety engine.  It is a deterministic review
    router for common oncology supplement scenarios, designed to tell the
    patient "ask your oncology team/pharmacist" instead of giving advice.
    """

    haystack = _normalize(" ".join([new_medication or "", notes or "", *[m or "" for m in current_medications]]))
    new_text = _normalize(f"{new_medication or ''} {notes or ''}")
    current_text = _normalize(" ".join(current_medications or ()))
    flags = []
    for rule in RULES:
        trigger_hit = _contains_any(new_text, rule.trigger_terms)
        context_hit = _contains_any(current_text, rule.context_terms) or _contains_any(haystack, rule.context_terms)
        if trigger_hit and (context_hit or not rule.context_terms):
            flags.append({
                "rule_id": rule.id,
                "severity": rule.severity,
                "message": rule.message,
                "clinician_action": rule.clinician_action,
                "matched_trigger_terms": [term for term in rule.trigger_terms if _normalize(term) in new_text],
                "matched_context_terms": [term for term in rule.context_terms if _normalize(term) in haystack],
            })
    return {
        "checker_version": INTERACTION_CHECKER_VERSION,
        "status": "review_needed" if flags else "no_specific_rule_hit",
        "flags": flags,
        "claim_boundary": (
            "Deterministic supplement/medication review routing only. It is incomplete and does not determine "
            "whether a medication or supplement is safe. Oncology-team/pharmacist review remains required."
        ),
    }


def _normalize(text: str) -> str:
    text = text.lower().replace("’", "'").replace("'", "").replace(".", " ")
    text = re.sub(r"[^a-z0-9+/ -]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(_normalize(term) in text for term in terms)


__all__ = ["check_medication_interactions", "INTERACTION_CHECKER_VERSION"]
