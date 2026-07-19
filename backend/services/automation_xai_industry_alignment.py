"""Industry-alignment roadmap for NLCare automation and patient XAI.

This artifact is intentionally governance-oriented. It describes controls that
make automation and explanations more reviewable, but it does not enable real
clinical notification channels or promote any model output.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


OUTPUT_PATH = Path("Data/evals/governance/latest_automation_xai_industry_alignment.json")
DOC_PATH = Path("docs/automation_xai_industry_alignment.md")

AUTOMATION_DOSSIER_PATH = Path("Data/evals/ops/latest_automation_reliability_dossier.json")
PATIENT_XAI_DOSSIER_PATH = Path("Data/evals/governance/latest_patient_xai_readability_dossier.json")
CITATION_WINDOW_PATH = Path("Data/evals/rag/latest_citation_window_sensitivity.json")

CLAIM_BOUNDARY = (
    "This roadmap is engineering governance only. It is not clinical validation, not clinician sign-off, "
    "not emergency coverage, not proof of patient benefit, and not healthcare production readiness."
)

AUTOMATION_CONTROLS: tuple[dict[str, Any], ...] = (
    {
        "id": "outbox_first_source_of_truth",
        "industry_reason": "External channels fail; the system needs a local auditable queue before dispatch.",
        "minimum_requirement": "Every high-risk event creates a local alert row before any webhook/email/SMS/Viber attempt.",
        "current_status": "implemented_as_engineering_preview",
    },
    {
        "id": "redacted_signed_event_envelope",
        "industry_reason": "Webhook payloads should not leak raw chat text or patient identifiers.",
        "minimum_requirement": "Use redacted payloads, HMAC signatures, timestamp tolerance, and replay protection.",
        "current_status": "implemented_as_engineering_preview",
    },
    {
        "id": "idempotency_and_deduplication",
        "industry_reason": "Retries must not create duplicate clinical-review tasks.",
        "minimum_requirement": "Stable event_id/idempotency_key across retries and dedupe checks in the receiver.",
        "current_status": "partially_documented_needs_ui_visibility",
    },
    {
        "id": "retry_dead_letter_and_requeue",
        "industry_reason": "Operators need to see failed notification attempts and recover safely.",
        "minimum_requirement": "Bounded retries, dead-letter reason, manual requeue, and no loss of local alert.",
        "current_status": "implemented_as_contract_needs_operator_ui",
    },
    {
        "id": "delivery_receipt_not_human_acknowledgement",
        "industry_reason": "Transport delivery is not the same as clinician review.",
        "minimum_requirement": "Separate delivery receipt, opened/reviewed, and manual acknowledgement states.",
        "current_status": "implemented_as_boundary_needs_dashboard_card",
    },
    {
        "id": "test_recipient_only_external_channels",
        "industry_reason": "Without clinical operations ownership, real alerting can create false assurance.",
        "minimum_requirement": "Email/SMS/Viber/n8n are disabled by default and limited to synthetic test recipients.",
        "current_status": "implemented",
    },
)

XAI_CONTROLS: tuple[dict[str, Any], ...] = (
    {
        "id": "explanation_contract_per_surface",
        "industry_reason": "Each patient-visible number needs meaning, calculation, uncertainty, and safe next step copy.",
        "minimum_requirement": "Expose a typed explanation envelope for every KPI/model/review-count surface.",
        "current_status": "specified_needs_api_contract",
    },
    {
        "id": "model_card_and_feature_dictionary",
        "industry_reason": "Reviewers need to know model version, inputs, missingness handling, and synthetic-only limits.",
        "minimum_requirement": "Attach model-card and feature-dictionary links to each synthetic ML output.",
        "current_status": "documented_needs_frontend_drawer",
    },
    {
        "id": "uncertainty_and_abstention_first",
        "industry_reason": "Low evidence should reduce confidence or abstain, not produce more decisive language.",
        "minimum_requirement": "Show modalities present/missing, abstention reason, confidence source, and known weakness.",
        "current_status": "partially_implemented",
    },
    {
        "id": "non_causal_feature_contributions",
        "industry_reason": "Feature importance can be mistaken as clinical causality.",
        "minimum_requirement": "Label contribution displays as non-causal synthetic engineering explanations.",
        "current_status": "planned",
    },
    {
        "id": "retrieval_grounding_visibility",
        "industry_reason": "RAG answers need visible evidence limits, especially when citation precision is weak.",
        "minimum_requirement": "Show answerability, citation support, source-tier policy, and unsupported-context warnings.",
        "current_status": "partially_implemented",
    },
    {
        "id": "negative_results_visible_to_reviewers",
        "industry_reason": "Credibility improves when failed experiments are not hidden.",
        "minimum_requirement": "Keep pruner regression, BM25 comparison, reranker non-proof, and held-out weaknesses visible.",
        "current_status": "implemented_as_governance_artifacts",
    },
)

BACKLOG: tuple[dict[str, Any], ...] = (
    {
        "rank": 1,
        "item": "Build an Automation Center admin card",
        "why": "Shows outbox state, test-recipient delivery receipts, retry/dead-letter status, and manual acknowledgement separately.",
        "side": "automation",
        "live_clinical_claim_allowed": False,
    },
    {
        "rank": 2,
        "item": "Add typed patient-XAI explanation envelopes",
        "why": "Turns every patient-visible number into meaning + calculation + uncertainty + allowed next action.",
        "side": "xai",
        "live_clinical_claim_allowed": False,
    },
    {
        "rank": 3,
        "item": "Add model-card and feature-dictionary drawers",
        "why": "Makes synthetic ML outputs auditable without making them look clinically authoritative.",
        "side": "xai",
        "live_clinical_claim_allowed": False,
    },
    {
        "rank": 4,
        "item": "Run generated-answer A/B before any citation-window change",
        "why": "A smaller citation window may improve precision but could reduce answer support; live behavior should not change from retrieval-only metrics.",
        "side": "rag_xai",
        "live_clinical_claim_allowed": False,
    },
    {
        "rank": 5,
        "item": "Create n8n inactive templates for escalation digest and dead-letter review",
        "why": "Industry-aligned automation can be demonstrated without enabling real patient alerting.",
        "side": "automation",
        "live_clinical_claim_allowed": False,
    },
)


def build_automation_xai_industry_alignment(
    *,
    output_path: str | Path = OUTPUT_PATH,
    doc_path: str | Path = DOC_PATH,
) -> dict[str, Any]:
    automation = _read_json(AUTOMATION_DOSSIER_PATH)
    xai = _read_json(PATIENT_XAI_DOSSIER_PATH)
    citation = _read_json(CITATION_WINDOW_PATH)
    evidence = {
        "automation_reliability_status": automation.get("status", "missing"),
        "patient_xai_readability_status": xai.get("status", "missing"),
        "citation_window_status": citation.get("status", "missing"),
        "citation_window_promotion_recommendation": citation.get("promotion_recommendation", "not_run"),
    }
    missing = [name for name, value in evidence.items() if value == "missing"]
    payload = {
        "schema_version": "automation_xai_industry_alignment_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if not missing else "needs_attention",
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "automation_live_delivery_enabled": False,
        "patient_benefit_claim": False,
        "diagnostic_authority_claim": False,
        "treatment_recommendation_claim": False,
        "real_emergency_coverage_claim": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_inputs": evidence,
        "missing_evidence_inputs": missing,
        "automation_controls": list(AUTOMATION_CONTROLS),
        "xai_controls": list(XAI_CONTROLS),
        "automation_control_count": len(AUTOMATION_CONTROLS),
        "xai_control_count": len(XAI_CONTROLS),
        "ranked_backlog": list(BACKLOG),
        "industry_alignment_summary": (
            "NLCare is closest to industry practice when automation is treated as an auditable local review workflow "
            "and XAI is treated as a typed explanation contract with uncertainty and known weaknesses visible."
        ),
        "what_is_still_not_industry_ready": [
            "No real clinical operations owner or on-call process.",
            "No real PHI channel review or compliance sign-off.",
            "No external clinician acknowledgement workflow.",
            "No real patient data or clinical validation.",
            "No live alerting beyond synthetic test-recipient scaffolding.",
        ],
    }
    _write_json(Path(output_path), payload)
    _write_doc(Path(doc_path), payload)
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_doc(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Automation and XAI Industry Alignment",
        "",
        CLAIM_BOUNDARY,
        "",
        "## Summary",
        "",
        f"- Status: `{payload['status']}`",
        f"- Automation live delivery enabled: `{payload['automation_live_delivery_enabled']}`",
        f"- Healthcare production ready: `{payload['healthcare_production_ready']}`",
        f"- Automation controls: `{payload['automation_control_count']}`",
        f"- XAI controls: `{payload['xai_control_count']}`",
        "",
        "## Automation Controls",
        "",
    ]
    for item in payload["automation_controls"]:
        lines.extend([
            f"### {item['id']}",
            "",
            f"- Reason: {item['industry_reason']}",
            f"- Minimum requirement: {item['minimum_requirement']}",
            f"- Current status: `{item['current_status']}`",
            "",
        ])
    lines.extend(["## XAI Controls", ""])
    for item in payload["xai_controls"]:
        lines.extend([
            f"### {item['id']}",
            "",
            f"- Reason: {item['industry_reason']}",
            f"- Minimum requirement: {item['minimum_requirement']}",
            f"- Current status: `{item['current_status']}`",
            "",
        ])
    lines.extend(["## Ranked Backlog", ""])
    for item in payload["ranked_backlog"]:
        lines.append(f"{item['rank']}. {item['item']} - {item['why']}")
    lines.extend([
        "",
        "## Still Not Industry Ready",
        "",
    ])
    for item in payload["what_is_still_not_industry_ready"]:
        lines.append(f"- {item}")
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = ["build_automation_xai_industry_alignment"]
