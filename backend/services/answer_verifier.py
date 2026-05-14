"""Deterministic post-generation verifier for medical support answers.

This is not a clinical judge. It is a final safety/grounding gate that catches
obvious violations before a reply is returned or cached.
"""

from __future__ import annotations

import re
from typing import Any


DIAGNOSTIC_PATTERNS = (
    r"\byou have (breast )?cancer\b",
    r"\byou do not have (breast )?cancer\b",
    r"\bmetastasis (is )?confirmed\b",
    r"\bprogression (is )?confirmed\b",
)

TREATMENT_DIRECTIVE_PATTERNS = (
    r"\byou should stop\b",
    r"\bstop your (chemo|chemotherapy|medication)\b",
    r"\bincrease your dose\b",
    r"\bdecrease your dose\b",
    r"\bskip your dose\b",
    r"\btake \d+ ?(mg|ml)\b",
)


REFUSAL_INTENTS_FOR_VERIFIER = frozenset({
    "safety_boundary",
    "treatment_decision_boundary",
    "security_boundary",
})


def verify_patient_support_answer(
    reply: str,
    *,
    citations: list[dict[str, Any]] | None = None,
    retrieved_context: list[dict[str, Any]] | None = None,
    safety: dict[str, Any] | None = None,
    intent: str | None = None,
) -> dict[str, Any]:
    reply = reply or ""
    lower = reply.lower()
    issues: list[str] = []

    if _matches_any(DIAGNOSTIC_PATTERNS, lower):
        issues.append("diagnostic_claim")
    if _matches_any(TREATMENT_DIRECTIVE_PATTERNS, lower):
        issues.append("treatment_directive")
    # Refusal intents intentionally strip citations on background context to
    # avoid the appearance of evidence-backed clinical guidance — see
    # generate_answer in agent_rag. Don't flag the deliberate absence here.
    if (
        retrieved_context
        and not citations
        and intent not in REFUSAL_INTENTS_FOR_VERIFIER
    ):
        issues.append("retrieved_context_without_citations")
    if (safety or {}).get("level") == "high_risk":
        if not any(term in lower for term in ("emergency", "clinician", "oncology", "care team")):
            issues.append("high_risk_missing_escalation")

    return {
        "status": "passed" if not issues else "failed",
        "issues": issues,
        "claim_boundary": "Deterministic verifier for obvious safety/citation failures only; not a medical correctness judge.",
    }


def safe_repair_reply(reply: str, verifier: dict[str, Any]) -> str:
    if verifier.get("status") == "passed":
        return reply
    issues = set(verifier.get("issues") or [])
    if {"diagnostic_claim", "treatment_directive"} & issues:
        return (
            "I cannot diagnose, confirm metastasis/progression, or decide treatment or medication changes. "
            "I can help organize the information for clinician review, and you should contact your oncology care team for personal guidance."
        )
    if "high_risk_missing_escalation" in issues:
        return (
            f"{reply} Please contact your oncology care team promptly, or emergency services for severe or sudden symptoms."
        )
    return reply


def _matches_any(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)
