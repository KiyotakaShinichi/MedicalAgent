"""Post-generation safety validator.

The deterministic safety gate runs **before** retrieval and generation; this
validator runs **after** the LLM produces a reply.  It re-reads what the
model is about to say and blocks output if the reply makes a claim that
the system is not allowed to make, even when the LLM decided to make it
anyway.

What it catches
---------------
  - **Diagnosis claims** — "you have cancer", "this is metastatic disease",
    "the lesion is malignant", etc.
  - **Treatment recommendations** — "you should take X", "increase the dose",
    "skip your next cycle", "switch to a different regimen".
  - **Prognosis estimates** — "you have N months", "survival rate is X%",
    "you'll likely die from this".
  - **Genetic-risk overclaims** — "you are BRCA-positive", "this VUS is
    positive", "your relatives will develop cancer".
  - **Tumor-marker overclaims** — "your cancer has returned" based on a
    standalone tumor-marker reading.
  - **Dosage instructions** — "take 10mg twice a day", etc.

Why a separate layer
--------------------
The pre-gen safety gate handles INPUT patterns ("can you tell me if I have
cancer?").  The post-gen validator handles OUTPUT patterns ("you have
cancer.").  They are complementary — both must run, and the validator is
the last line of defence before a patient or clinician sees the reply.

Action on detection: by default we mark the reply as ``blocked`` and the
caller substitutes a safe refusal.  The validator never mutates the reply
in-place; it returns a decision the caller acts on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from backend.services.dep001d_output_actionability import classify_output_actionability
from backend.services.medical_claim_boundary import classify_medical_claim


# ─── Pattern catalog ─────────────────────────────────────────────────────────
# Patterns are intentionally precise — they target *claims*, not discussion.
# A reply that says "diagnosis is determined by a clinician" must not trip
# the diagnosis rule, but "your diagnosis is X" must.  We rely on phrase
# context (subject pronouns, modal verbs) rather than bare keyword matching.

# Reuse the same RegEx flags everywhere so a future edit can't accidentally
# loosen one pattern by forgetting case-insensitivity.
_FLAGS = re.IGNORECASE | re.DOTALL


@dataclass(frozen=True)
class _Rule:
    """One validator rule.  ``patterns`` is a tuple of compiled regexes —
    any match short-circuits and surfaces the rule's ``code`` + ``severity``."""
    code: str
    severity: str  # "high" | "medium"
    description: str
    patterns: tuple[re.Pattern[str], ...]


def _compile(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, _FLAGS) for p in patterns)


# Patterns are written to catch the most common phrasing for each banned
# claim category.  The list is conservative — false positives are preferable
# to false negatives in a safety layer.

DIAGNOSIS_RULES = _Rule(
    code="diagnosis_claim",
    severity="high",
    description="Model is asserting a diagnosis rather than deferring to a clinician.",
    patterns=_compile(
        r"\byou\s+(do\s+|definitely\s+|likely\s+)?have\b(?!\s+(not|no)\b)\s+(cancer|breast cancer|a (malignant|metastatic)|tumor|metastases|recurrence)",
        r"\byou\s+(definitely\s+|certainly\s+)?have\s+(recurrent|metastatic|malignant)\s+(breast\s+)?cancer\b",
        r"\bthis\s+is\s+(definitely\s+)?(cancer|malignant|metastatic|a tumour|a tumor|recurrence)",
        r"\bthe\s+(lesion|mass|nodule)\s+is\s+(malignant|cancerous|metastatic)",
        r"\byou\s+are\s+diagnosed\s+with\b",
        r"\bI\s+(can\s+)?diagnose\s+you\b",
    ),
)

PRIVACY_DISCLOSURE_RULES = _Rule(
    code="privacy_or_cross_patient_disclosure",
    severity="high",
    description="Model is disclosing private or another patient's information.",
    patterns=_compile(
        r"\b(here (is|are)|I (found|retrieved|opened|exported))\b.{0,80}\b(private|confidential|protected|other patient|another patient)\b.{0,80}\b(record|chart|address|phone|email|lab|imaging|profile|data)\b",
        r"\b(other|another) patient'?s\b.{0,70}\b(record|chart|address|phone|email|lab|imaging|profile|data)\b",
    ),
)

SECURITY_DISCLOSURE_RULES = _Rule(
    code="system_or_policy_disclosure",
    severity="high",
    description="Model is exposing hidden instructions or bypassing an authority boundary.",
    patterns=_compile(
        r"\bhere (is|are)\b.{0,60}\b(hidden|internal|secret|system|developer)\b.{0,40}\b(instruction\w*|prompt\w*|rule\w*|polic(?:y|ies)|message\w*)\b",
        r"\b(clinician|doctor|care team|pharmacist|genetic counselor)\b.{0,35}\b(review|approval|oversight)\b.{0,25}\b(is|are)?\s*(unnecessary|not needed|optional)\b",
    ),
)

TREATMENT_RULES = _Rule(
    code="treatment_recommendation",
    severity="high",
    description="Model is recommending a treatment change or new medication.",
    patterns=_compile(
        r"\byou\s+should\s+(take|start|stop|switch to|change to|increase|decrease|skip)\b",
        r"\bI\s+recommend\s+(you\s+)?(take|start|stop|switch|change|use|try)\b",
        r"\b(start|stop|skip|change|increase|reduce|switch)\s+(your\s+)?(chemo|chemotherapy|cycle|dose|medication|treatment)\b",
        r"\b(begin|start)\s+taking\b",
    ),
)

PROGNOSIS_RULES = _Rule(
    code="prognosis_estimate",
    severity="high",
    description="Model is estimating survival, life expectancy, or definitive course.",
    patterns=_compile(
        r"\byou\s+have\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|a few|several)\s+(months?|weeks?|years?)\s+(to\s+live|left)",
        # "Survival rate is 85%", "survival rate of 85%", "life expectancy is 50 percent" — any
        # combination of the rate/is/of glue between "survival|life expectancy" and a percentage.
        r"\b(survival|life\s+expectancy)\s+(rate\s+)?(is\s+)?(of\s+)?\d{1,3}\s*(percent|%)",
        r"\byou\s+(will|are\s+going\s+to)\s+(die|likely die|pass away)\b",
        r"\b\d{1,3}\s*(percent|%)\s+chance\s+(of\s+)?(survival|surviving|dying|recurrence|cure)",
    ),
)

DOSAGE_RULES = _Rule(
    code="dosage_instruction",
    severity="high",
    description="Model is providing a specific dosage / schedule instruction.",
    patterns=_compile(
        r"\btake\s+\d+\s*(mg|milligrams|grams|g|ml|tablets?|capsules?)\b",
        r"\b\d+\s*(mg|milligrams)\s+(twice|three\s+times|four\s+times)\s+(a\s+)?day\b",
        r"\b(once|twice|thrice|three\s+times)\s+(a\s+)?day\s+for\s+\d+\s+(days?|weeks?)\b",
    ),
)

GENETICS_OVERCLAIM_RULES = _Rule(
    code="genetic_risk_overclaim",
    severity="high",
    description="Model is asserting a positive genetic finding from a VUS or unclear result.",
    patterns=_compile(
        r"\bVUS\s+(means|is)\s+(you\s+are\s+)?(positive|cancer|definitely)\b",
        r"\byou\s+are\s+(BRCA[12]?-?|HER2-?|ER-?|PR-?)\s*positive\b(?!\s+for\b)",
        r"\byour\s+(family|relatives|children|siblings)\s+(will|are going to)\s+(develop|get|have)\s+cancer\b",
    ),
)

TUMOR_MARKER_RULES = _Rule(
    code="tumor_marker_overclaim",
    severity="high",
    description="Model is interpreting a standalone tumor marker as recurrence / progression.",
    patterns=_compile(
        r"\b(elevated|high|rising)\s+(CA\s*15-?3|CA\s*27\.?29|CEA)\s+(means|indicates|shows)\s+(cancer|recurrence|progression|metastasis)",
        r"\b(CA\s*15-?3|CA\s*27\.?29|CEA)\s+(proves|confirms|means|shows|indicates)\s+"
        r"(?:that\s+)?(?:the\s+)?(recurrence|progression|metastasis|cancer\s+(has\s+)?returned)\b",
        r"\b(?:a\s+|the\s+|your\s+)?tumou?r\s+markers?\s+"
        r"(?:alone\s+)?(?:proves?|confirms?|means|shows|indicates)\s+"
        r"(?:that\s+)?(?:the\s+|your\s+)?(?:recurrence|progression|metastasis|cancer\s+(?:has\s+)?returned)\b",
        r"\byour\s+cancer\s+(has\s+)?(come back|returned|recurred|is back)\s+(because|based on|due to)\s+.*?(marker|CA\s*15|CA\s*27|CEA)",
    ),
)


ALL_RULES: tuple[_Rule, ...] = (
    DIAGNOSIS_RULES,
    PRIVACY_DISCLOSURE_RULES,
    SECURITY_DISCLOSURE_RULES,
    TREATMENT_RULES,
    PROGNOSIS_RULES,
    DOSAGE_RULES,
    GENETICS_OVERCLAIM_RULES,
    TUMOR_MARKER_RULES,
)


# ─── Decision envelope ───────────────────────────────────────────────────────


@dataclass
class ValidatorDecision:
    """One validator pass over one reply."""

    decision: str            # "allowed" | "blocked"
    triggered_rules: list[str] = field(default_factory=list)
    matched_excerpts: list[dict[str, str]] = field(default_factory=list)
    suggested_response: str | None = None  # safe replacement when blocked
    severity: str = "low"    # "low" when allowed, otherwise highest rule
    claim_boundary: dict | None = None
    semantic_actionability: dict | None = None

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "triggered_rules": list(self.triggered_rules),
            "matched_excerpts": list(self.matched_excerpts),
            "suggested_response": self.suggested_response,
            "severity": self.severity,
            "claim_boundary": self.claim_boundary,
            "semantic_actionability": self.semantic_actionability,
        }


DEFAULT_REFUSAL = (
    "I'm not able to share that as a clinical interpretation. "
    "Please bring this up with your oncology team so they can review the "
    "full context. If anything feels urgent — fever after chemotherapy, "
    "heavy bleeding, severe pain, or trouble breathing — contact your "
    "care team or local emergency services immediately."
)


def validate_reply(
    reply: str,
    *,
    rules: Iterable[_Rule] = ALL_RULES,
    safe_refusal: str = DEFAULT_REFUSAL,
) -> ValidatorDecision:
    """Re-read an LLM reply and decide whether to allow or block it.

    Returns ``ValidatorDecision(decision="allowed", ...)`` when no rule
    fires; ``ValidatorDecision(decision="blocked", ..., suggested_response=safe_refusal)``
    when at least one rule fires.  The caller is responsible for substituting
    the suggested response into the user-facing payload.
    """
    if not isinstance(reply, str) or not reply.strip():
        return ValidatorDecision(
            decision="blocked",
            triggered_rules=["malformed_or_empty_output"],
            matched_excerpts=[],
            suggested_response=safe_refusal,
            severity="high",
            claim_boundary=classify_medical_claim(""),
            semantic_actionability={
                "decision": "blocked",
                "reason": "malformed_or_empty_output",
                "failure_reason": "malformed_or_empty_output",
            },
        )

    claim_boundary = classify_medical_claim(reply)
    try:
        semantic_actionability = classify_output_actionability(reply).to_dict()
    except Exception as exc:
        semantic_actionability = {
            "decision": "blocked",
            "blocked": True,
            "reason": "output_actionability_validation_unavailable",
            "failure_reason": f"validator_exception:{type(exc).__name__}",
        }

    triggered: list[str] = []
    excerpts: list[dict[str, str]] = []
    severity = "low"

    for rule in rules:
        for pattern in rule.patterns:
            match = pattern.search(reply)
            if match:
                triggered.append(rule.code)
                excerpts.append({
                    "rule": rule.code,
                    "excerpt": _excerpt(reply, match.start(), match.end()),
                })
                # Track highest severity seen so far.
                if rule.severity == "high":
                    severity = "high"
                elif rule.severity == "medium" and severity != "high":
                    severity = "medium"
                # Don't break — surface every matching rule, not just the first.
                break

    if not triggered:
        if claim_boundary.get("decision") == "blocked":
            triggered = list(claim_boundary.get("blocked_claim_types") or ["medical_claim_boundary"])
            return ValidatorDecision(
                decision="blocked",
                triggered_rules=triggered,
                matched_excerpts=[
                    {"rule": item["claim_type"], "excerpt": item["excerpt"]}
                    for item in claim_boundary.get("matched_claims", [])
                    if item.get("decision") == "blocked"
                ],
                suggested_response=safe_refusal,
                severity="high",
                claim_boundary=claim_boundary,
                semantic_actionability=semantic_actionability,
            )
        if semantic_actionability["decision"] == "blocked":
            return ValidatorDecision(
                decision="blocked",
                triggered_rules=["semantic_output_actionability"],
                matched_excerpts=[],
                suggested_response=safe_refusal,
                severity="high",
                claim_boundary=claim_boundary,
                semantic_actionability=semantic_actionability,
            )
        return ValidatorDecision(
            decision="allowed",
            claim_boundary=claim_boundary,
            semantic_actionability=semantic_actionability,
        )

    return ValidatorDecision(
        decision="blocked",
        triggered_rules=triggered,
        matched_excerpts=excerpts,
        suggested_response=safe_refusal,
        severity=severity,
        claim_boundary=claim_boundary,
        semantic_actionability=semantic_actionability,
    )


def _excerpt(text: str, start: int, end: int, window: int = 60) -> str:
    """A short window around the match to help an auditor see *why* it fired
    without dumping the whole reply."""
    a = max(0, start - window)
    b = min(len(text), end + window)
    return text[a:b].strip()


__all__ = [
    "ALL_RULES",
    "DEFAULT_REFUSAL",
    "ValidatorDecision",
    "validate_reply",
]
