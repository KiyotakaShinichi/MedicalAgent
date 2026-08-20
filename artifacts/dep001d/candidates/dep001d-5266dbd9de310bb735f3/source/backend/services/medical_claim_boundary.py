"""Medical claim-boundary classifier.

This is the pre-output policy classifier for NLCare replies. It labels the
kind of medical claim a draft answer is making, then decides whether that
claim type is allowed, must be hedged, or must be blocked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


CLAIM_BOUNDARY_VERSION = "medical_claim_boundary_v1_2026_05"
FLAGS = re.IGNORECASE | re.DOTALL


@dataclass(frozen=True)
class ClaimType:
    code: str
    decision: str  # allowed | caution | blocked
    description: str
    patterns: tuple[re.Pattern[str], ...] = field(default_factory=tuple)


def _compile(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern, FLAGS) for pattern in patterns)


CLAIM_TYPES: tuple[ClaimType, ...] = (
    ClaimType(
        code="general_education",
        decision="allowed",
        description="General educational explanation without patient-specific decision.",
        patterns=_compile(r"\b(generally|in general|can mean|stands for|refers to)\b"),
    ),
    ClaimType(
        code="record_organization",
        decision="allowed",
        description="Organizes user-provided data without interpreting it clinically.",
        patterns=_compile(r"\b(I'?ve noted|I can help record|logged|save this|organize)\b"),
    ),
    ClaimType(
        code="monitoring_signal",
        decision="caution",
        description="Monitor-only signal that requires clinician review and uncertainty language.",
        patterns=_compile(r"\b(monitoring signal|pattern|trend|review with|care team)\b"),
    ),
    ClaimType(
        code="diagnosis_claim",
        decision="blocked",
        description="Asserts diagnosis, malignancy, metastasis, recurrence, or cancer status.",
        patterns=_compile(
            r"\byou\s+(do\s+|definitely\s+|likely\s+)?have\b(?!\s+(not|no)\b)\s+(cancer|breast cancer|metastatic|recurrence)",
            r"\bthis\s+is\s+(definitely\s+)?(cancer|malignant|metastatic|recurrence)",
            r"\bthe\s+(lesion|mass|nodule)\s+is\s+(malignant|cancerous|metastatic)",
        ),
    ),
    ClaimType(
        code="treatment_recommendation",
        decision="blocked",
        description="Recommends starting/stopping/changing treatment or medications.",
        patterns=_compile(
            r"\byou\s+should\s+(take|start|stop|switch|change|increase|decrease|skip)\b",
            r"\bI\s+recommend\s+(you\s+)?(take|start|stop|switch|change|use|try)\b",
        ),
    ),
    ClaimType(
        code="dosage_instruction",
        decision="blocked",
        description="Gives specific dosage or medication schedule.",
        patterns=_compile(r"\btake\s+\d+\s*(mg|milligrams|grams|g|ml|tablets?|capsules?)\b"),
    ),
    ClaimType(
        code="prognosis_estimate",
        decision="blocked",
        description="Predicts survival, life expectancy, cure chance, or recurrence probability.",
        patterns=_compile(
            r"\byou\s+have\s+\d+\s+(months?|weeks?|years?)\s+(to\s+live|left)",
            r"\b\d{1,3}\s*(percent|%)\s+chance\s+(of\s+)?(survival|recurrence|cure|dying)",
        ),
    ),
    ClaimType(
        code="genetic_risk_overclaim",
        decision="blocked",
        description="Treats VUS or unclear genetic data as a confirmed genetic diagnosis/risk.",
        patterns=_compile(
            r"\bVUS\s+(means|is)\s+(you\s+are\s+)?(positive|definitely)",
            r"\byour\s+(family|relatives|children|siblings)\s+(will|are going to)\s+(develop|get|have)\s+cancer",
        ),
    ),
    ClaimType(
        code="tumor_marker_overclaim",
        decision="blocked",
        description="Uses tumor marker value alone to claim recurrence/progression.",
        patterns=_compile(
            r"\b(elevated|high|rising)\s+(CA\s*15-?3|CA\s*27\.?29|CEA)\s+(means|indicates|shows)\s+(cancer|recurrence|progression|metastasis)",
            r"\byour\s+cancer\s+(has\s+)?(come back|returned|recurred|is back)\s+(because|based on|due to)\s+.*?(marker|CA\s*15|CA\s*27|CEA)",
        ),
    ),
    ClaimType(
        code="false_reassurance",
        decision="blocked",
        description="Tells a patient a symptom/supplement/result is safe/fine or not worth worry without clinician review.",
        patterns=_compile(
            r"\b(no need to worry|nothing to worry about|you are fine|it is fine|this is safe)\b",
            r"\b(safe|fine|okay)\s+(with|during)\s+(chemo|chemotherapy|radiation|treatment)\b",
        ),
    ),
    ClaimType(
        code="pregnancy_pediatric_boundary",
        decision="caution",
        description="Pregnancy, breastfeeding, fertility, or minor/pediatric context requires clinician-specific review.",
        patterns=_compile(
            r"\b(pregnant|pregnancy|breastfeeding|breast feeding|fertility|trying to conceive)\b",
            r"\b(child|minor|pediatric|paediatric|teenager|under\s*18)\b",
        ),
    ),
    ClaimType(
        code="survivorship_support",
        decision="caution",
        description="Survivorship, recurrence anxiety, palliative/supportive-care questions should stay educational and route to care team when patient-specific.",
        patterns=_compile(
            r"\b(survivorship|after treatment|surveillance|recurrence anxiety|palliative|supportive care)\b",
        ),
    ),
)


def classify_medical_claim(text: str) -> dict[str, Any]:
    matched: list[dict[str, str]] = []
    decisions: list[str] = []
    for claim_type in CLAIM_TYPES:
        for pattern in claim_type.patterns:
            match = pattern.search(text or "")
            if match:
                matched.append({
                    "claim_type": claim_type.code,
                    "decision": claim_type.decision,
                    "description": claim_type.description,
                    "excerpt": _excerpt(text, match.start(), match.end()),
                })
                decisions.append(claim_type.decision)
                break

    if "blocked" in decisions:
        decision = "blocked"
    elif "caution" in decisions:
        decision = "caution"
    else:
        decision = "allowed"
    return {
        "version": CLAIM_BOUNDARY_VERSION,
        "decision": decision,
        "matched_claims": matched,
        "blocked_claim_types": [
            item["claim_type"] for item in matched if item["decision"] == "blocked"
        ],
        "claim_boundary": (
            "Allowed outputs may educate or organize records. Blocked outputs "
            "include diagnosis, treatment recommendation, dosage, prognosis, "
            "genetic-risk overclaim, tumor-marker overclaim, and false reassurance."
        ),
    }


def claim_boundary_manifest() -> dict[str, Any]:
    return {
        "version": CLAIM_BOUNDARY_VERSION,
        "claim_types": [
            {
                "code": claim.code,
                "decision": claim.decision,
                "description": claim.description,
            }
            for claim in CLAIM_TYPES
        ],
    }


def _excerpt(text: str, start: int, end: int, window: int = 60) -> str:
    return (text or "")[max(0, start - window):min(len(text or ""), end + window)].strip()


__all__ = [
    "CLAIM_BOUNDARY_VERSION",
    "CLAIM_TYPES",
    "claim_boundary_manifest",
    "classify_medical_claim",
]
