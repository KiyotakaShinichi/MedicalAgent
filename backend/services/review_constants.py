"""Shared enumerations for clinician review decisions, risk levels, and event types.

This module is the single Python source of truth for the strings exposed
on the API. The TypeScript frontend mirrors these in
``frontend-react/src/lib/constants.ts``; please update both when changing values.
"""

from __future__ import annotations


# ─── Clinician review decisions ──────────────────────────────────────────────
# Matches VALID_REVIEW_DECISIONS in backend/services/clinician_feedback.py.
REVIEW_DECISIONS: tuple[str, ...] = (
    "approved",
    "edited",
    "rejected",
    "unsafe",
    "missing_evidence",
    "wrong_escalation",
    "needs_followup",
)

REVIEW_TARGETS: tuple[str, ...] = ("summary", "risk_flag", "timeline_event")


REVIEW_REASON_CATEGORIES: tuple[str, ...] = (
    "evidence_quality",
    "missing_imaging_followup",
    "wrong_severity",
    "wrong_escalation_level",
    "incomplete_summary",
    "diagnostic_overreach",
    "treatment_overreach",
    "privacy_concern",
    "prompt_injection_concern",
    "other",
)


# ─── Risk severity levels (used by risk_engine.py) ──────────────────────────
RISK_LEVELS: tuple[str, ...] = ("info", "watch", "urgent_review")


# ─── Confidence levels (used by uncertainty layer) ──────────────────────────
CONFIDENCE_LEVELS: tuple[str, ...] = ("low", "moderate", "high")


# ─── Timeline / event types ─────────────────────────────────────────────────
EVENT_TYPES: tuple[str, ...] = (
    "lab_result",
    "symptom_report",
    "medication_log",
    "treatment_cycle",
    "imaging_report",
    "ai_risk_flag",
    "support_chat",
    "clinician_review",
    "upload",
    "audit_event",
)


# ─── Artifact freshness statuses ────────────────────────────────────────────
ARTIFACT_STATUSES: tuple[str, ...] = (
    "available",
    "not_generated",
    "stale",
    "error",
    "unavailable",
)


# ─── Safety / refusal route categories (from agent_rag.py intents) ──────────
SAFETY_REFUSAL_TYPES: tuple[str, ...] = (
    "diagnosis_refusal",
    "treatment_refusal",
    "medication_refusal",
    "security_block",
    "urgent_escalation",
    "insufficient_evidence_refusal",
    "allowed_support",
)
