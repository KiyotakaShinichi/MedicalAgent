/**
 * Shared MedicalAgent enums — mirror of backend/services/review_constants.py.
 *
 * When changing values here, update the Python module too. The strings are
 * what the API actually sends and accepts, so divergence is a real bug.
 */

// ── Clinician review decisions ─────────────────────────────────────────────
export const REVIEW_DECISIONS = [
  "approved",
  "edited",
  "rejected",
  "unsafe",
  "missing_evidence",
  "wrong_escalation",
  "needs_followup",
] as const;

export type ReviewDecision = (typeof REVIEW_DECISIONS)[number];

export const REVIEW_DECISION_LABELS: Record<ReviewDecision, string> = {
  approved: "Approve",
  edited: "Save edit",
  rejected: "Reject",
  unsafe: "Mark unsafe",
  missing_evidence: "Request more evidence",
  wrong_escalation: "Wrong escalation level",
  needs_followup: "Needs follow-up",
};

// ── Review targets ─────────────────────────────────────────────────────────
export const REVIEW_TARGETS = ["summary", "risk_flag", "timeline_event"] as const;
export type ReviewTarget = (typeof REVIEW_TARGETS)[number];

// ── Reason categories for rejected / edited / unsafe reviews ───────────────
export const REVIEW_REASON_CATEGORIES = [
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
] as const;
export type ReviewReasonCategory = (typeof REVIEW_REASON_CATEGORIES)[number];

export const REVIEW_REASON_LABELS: Record<ReviewReasonCategory, string> = {
  evidence_quality: "Evidence quality issue",
  missing_imaging_followup: "Missing imaging follow-up",
  wrong_severity: "Wrong severity",
  wrong_escalation_level: "Wrong escalation level",
  incomplete_summary: "Incomplete summary",
  diagnostic_overreach: "Diagnostic overreach",
  treatment_overreach: "Treatment overreach",
  privacy_concern: "Privacy concern",
  prompt_injection_concern: "Prompt injection concern",
  other: "Other",
};

// ── Risk severity levels ───────────────────────────────────────────────────
export const RISK_LEVELS = ["info", "watch", "urgent_review"] as const;
export type RiskLevel = (typeof RISK_LEVELS)[number];

export const RISK_LEVEL_LABELS: Record<RiskLevel, string> = {
  info: "Info",
  watch: "Watch",
  urgent_review: "Urgent — needs review",
};

// ── Confidence levels for uncertainty layer ────────────────────────────────
export const CONFIDENCE_LEVELS = ["low", "moderate", "high"] as const;
export type ConfidenceLevel = (typeof CONFIDENCE_LEVELS)[number];

// ── Artifact freshness statuses ────────────────────────────────────────────
export const ARTIFACT_STATUSES = [
  "available",
  "not_generated",
  "stale",
  "error",
  "unavailable",
] as const;
export type ArtifactStatus = (typeof ARTIFACT_STATUSES)[number];

// ── Safe-language phrases the UI uses instead of diagnostic language ───────
export const SAFETY_DISCLAIMER_SHORT =
  "Not a diagnosis. For clinician review only.";

export const SAFETY_DISCLAIMER_LONG =
  "MedicalAgent organizes treatment-monitoring evidence for clinician review. " +
  "It does not diagnose cancer, recommend treatment, or replace clinician judgment. " +
  "Call your care team or local emergency services if symptoms feel urgent.";

export const CLINICIAN_REVIEW_REQUIRED_LABEL = "Clinician review required";
export const AI_GENERATED_LABEL = "AI-generated";
