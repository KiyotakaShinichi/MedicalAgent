from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from backend.services.n8n_automation_templates import BLOCKED_PAYLOAD_FIELDS
from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_OUTPUT_PATH = "Data/evals/ops/latest_automation_reliability_dossier.json"
DEFAULT_DOC_PATH = "docs/automation_reliability_dossier.md"

CLAIM_BOUNDARY = (
    "Automation reliability evidence is engineering evidence only. It does not prove emergency coverage, "
    "clinician receipt, patient benefit, clinical validation, compliance, or healthcare production readiness. "
    "All external notification channels remain redacted, test-recipient-only unless an operator deliberately "
    "configures them under a separate security and clinical-review process."
)

REQUIRED_AUTOMATION_INVARIANTS: tuple[dict[str, Any], ...] = (
    {
        "id": "local_outbox_first",
        "description": "High-risk conversation events are written to the local review outbox before external dispatch.",
        "evidence_path": "Data/evals/safety/latest_high_risk_conversation_alert_eval.json",
        "required": True,
    },
    {
        "id": "redacted_signed_webhook",
        "description": "n8n/webhook events use signed redacted envelopes and block PHI-like payload fields.",
        "evidence_path": "Data/evals/ops/latest_n8n_signed_dispatch_eval.json",
        "required": True,
    },
    {
        "id": "inactive_import_templates",
        "description": "n8n templates are inactive by default and are optional scaffolds, not live clinical automation.",
        "evidence_path": "Data/evals/ops/latest_n8n_workflow_templates.json",
        "required": True,
    },
    {
        "id": "test_recipient_only_high_risk_delivery",
        "description": "High-risk review alerts require synthetic test-recipient mode when external dispatch is enabled.",
        "evidence_path": "Data/evals/safety/latest_high_risk_conversation_alert_eval.json",
        "required": True,
    },
    {
        "id": "delivery_receipt_not_acknowledgement",
        "description": "Channel delivery receipts are explicitly not treated as clinician acknowledgement.",
        "evidence_path": "Data/evals/safety/latest_high_risk_conversation_alert_eval.json",
        "required": True,
    },
    {
        "id": "retry_dead_letter_contract",
        "description": "Failed delivery attempts are retried with bounded attempts and then dead-lettered without losing the local alert.",
        "evidence_path": "Data/evals/safety/latest_high_risk_conversation_alert_eval.json",
        "required": True,
    },
    {
        "id": "preview_only_schedule_plan",
        "description": "Scheduled automation is documented as ready for scheduler/n8n, but no host scheduler is installed automatically.",
        "evidence_path": "Data/evals/ops/latest_automation_schedule_plan.json",
        "required": True,
    },
    {
        "id": "dry_run_control_plane",
        "description": "The automation control plane queues redacted engineering jobs in dry-run mode and sends no webhooks while building artifacts.",
        "evidence_path": "Data/evals/ops/latest_automation_control_plane.json",
        "required": True,
    },
)

CHANNEL_MATRIX: tuple[dict[str, Any], ...] = (
    {
        "channel": "email",
        "intended_use": "Notify an internal demo/reviewer inbox that a redacted review item exists.",
        "allowed_payload": ["alert_id", "priority", "review_path", "delivery_scope", "recipient_scope"],
        "blocked_payload": ["patient identifiers", "raw chat text", "diagnosis/treatment/prognosis language"],
        "minimum_controls": ["HMAC signature", "test recipient mode", "delivery receipt callback", "local outbox fallback"],
        "live_patient_status": "disabled_by_default",
    },
    {
        "channel": "sms",
        "intended_use": "Optional short redacted engineering notification to a test maintainer number.",
        "allowed_payload": ["alert_id", "priority", "sign-in-required review link label"],
        "blocked_payload": ["patient identifiers", "symptom details", "medical instructions"],
        "minimum_controls": ["test recipient mode", "rate limit", "delivery receipt callback", "dead-letter tracking"],
        "live_patient_status": "disabled_by_default",
    },
    {
        "channel": "viber_or_chatops",
        "intended_use": "Optional internal demo channel notification for reviewer workflow visibility.",
        "allowed_payload": ["alert_id", "priority", "workflow status", "review path"],
        "blocked_payload": ["patient identifiers", "raw transcript", "PHI", "clinical advice", "private traces"],
        "minimum_controls": ["private channel", "HMAC signature", "redaction check", "manual acknowledgement in NLCare"],
        "live_patient_status": "disabled_by_default",
    },
    {
        "channel": "admin_dashboard",
        "intended_use": "Primary source of truth for queued/notified/acknowledged review items.",
        "allowed_payload": ["local alert metadata", "delivery state", "attempt counts", "acknowledgement status"],
        "blocked_payload": ["clinical decision automation", "automatic patient messaging"],
        "minimum_controls": ["role-gated access", "local audit trail", "manual acknowledgement", "claim-boundary text"],
        "live_patient_status": "local_demo_source_of_truth",
    },
)

AUTOMATION_CENTER_VISIBILITY_REQUIREMENTS: tuple[dict[str, Any], ...] = (
    {
        "id": "local_outbox_status",
        "label": "Local outbox status",
        "must_show": ["queued", "notified", "dead_lettered", "manual_acknowledgement"],
        "why": "The dashboard remains the source of truth even when external channels are configured.",
    },
    {
        "id": "delivery_receipt_status",
        "label": "Delivery receipt status",
        "must_show": ["not_sent", "sent_to_test_recipient", "delivery_receipt_validated", "delivery_failed"],
        "why": "Delivery status is transport evidence only and must not be confused with clinician acknowledgement.",
    },
    {
        "id": "retry_dead_letter_visibility",
        "label": "Retry/dead-letter visibility",
        "must_show": ["attempt_count", "last_error", "next_retry_at", "dead_letter_reason"],
        "why": "Operators need failure visibility without losing the local review item.",
    },
    {
        "id": "redaction_and_signature_visibility",
        "label": "Redaction/signature visibility",
        "must_show": ["blocked_payload_fields", "signature_status", "replay_protection_status"],
        "why": "Webhook safety should be inspectable before any optional n8n/email/SMS/Viber path is used.",
    },
    {
        "id": "claim_boundary_visibility",
        "label": "Claim-boundary visibility",
        "must_show": ["not_emergency_service", "not_clinician_acknowledgement", "test_recipient_only"],
        "why": "Automation must not make the product look clinically monitored or production-ready.",
    },
)


def build_automation_reliability_dossier(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    doc_path: str | Path = DEFAULT_DOC_PATH,
) -> dict[str, Any]:
    artifacts = {
        "n8n_templates": _read_json("Data/evals/ops/latest_n8n_workflow_templates.json"),
        "signed_dispatch": _read_json("Data/evals/ops/latest_n8n_signed_dispatch_eval.json"),
        "automation_schedule": _read_json("Data/evals/ops/latest_automation_schedule_plan.json"),
        "automation_control_plane": _read_json("Data/evals/ops/latest_automation_control_plane.json"),
        "high_risk_alert_eval": _read_json("Data/evals/safety/latest_high_risk_conversation_alert_eval.json"),
    }

    checks = _evaluate_checks(artifacts)
    passed_count = sum(1 for item in checks if item["passed"])
    required_failed = [item for item in checks if item["required"] and not item["passed"]]
    status = "strong" if not required_failed and passed_count == len(checks) else "needs_attention"
    payload = {
        "schema_version": "automation_reliability_dossier_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "hipaa_compliance_claim": False,
        "phi_allowed": False,
        "live_patient_route_enabled": False,
        "external_delivery_enabled_by_default": False,
        "real_emergency_coverage_claim": False,
        "check_count": len(checks),
        "passed_count": passed_count,
        "failed_required_count": len(required_failed),
        "checks": checks,
        "channel_matrix": list(CHANNEL_MATRIX),
        "automation_center_visibility_requirements": list(AUTOMATION_CENTER_VISIBILITY_REQUIREMENTS),
        "automation_center_requirement_count": len(AUTOMATION_CENTER_VISIBILITY_REQUIREMENTS),
        "blocked_payload_fields": list(BLOCKED_PAYLOAD_FIELDS),
        "automation_maturity": {
            "level": "engineering_preview_strong" if status == "strong" else "engineering_preview_needs_attention",
            "why_not_higher": [
                "No real clinical operations team is configured.",
                "No real patient notification or emergency dispatch is permitted.",
                "No compliance review, incident-response SLA, or clinician on-call rota exists.",
                "External channels are optional scaffolds and disabled by default.",
            ],
        },
        "recommended_operator_steps_before_real_use": [
            "Complete security and privacy review for any real notification channel.",
            "Define on-call ownership, escalation hours, and failure handling outside the app.",
            "Obtain clinician workflow review and written operating procedure.",
            "Keep PHI out of webhook payloads; use sign-in-required dashboard links only.",
            "Run receipt and dead-letter drills before enabling any non-local notification.",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(_resolve(output_path), payload)
    _write_doc(_resolve(doc_path), payload)
    return payload


def _evaluate_checks(artifacts: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    high = artifacts["high_risk_alert_eval"]
    signed = artifacts["signed_dispatch"]
    templates = artifacts["n8n_templates"]
    schedule = artifacts["automation_schedule"]
    control = artifacts["automation_control_plane"]
    values = {
        "local_outbox_first": high.get("pass_rate") == 1.0 and high.get("case_count", 0) >= 12,
        "redacted_signed_webhook": (
            signed.get("status") == "strong"
            and signed.get("phi_allowed") is False
            and signed.get("network_request_sent") is False
            and signed.get("signature_valid") is True
            and signed.get("nested_blocked_field_caught") is True
        ),
        "inactive_import_templates": (
            templates.get("status") == "ready_for_optional_import"
            and templates.get("template_count", 0) >= 10
            and templates.get("phi_allowed") is False
            and templates.get("live_patient_route_enabled") is False
        ),
        "test_recipient_only_high_risk_delivery": high.get("synthetic_test_recipient_enforced") is True,
        "delivery_receipt_not_acknowledgement": high.get("delivery_receipt_distinct_from_clinician_acknowledgement") is True,
        "retry_dead_letter_contract": high.get("retry_dead_letter_contract_tested") is True,
        "preview_only_schedule_plan": (
            schedule.get("status") == "ready_for_scheduler_or_n8n"
            and schedule.get("scheduler_installed") is False
            and schedule.get("phi_allowed") is False
        ),
        "dry_run_control_plane": (
            control.get("status") == "strong"
            and control.get("commands_executed") is False
            and control.get("webhooks_sent") is False
            and control.get("accepted_event_count") == control.get("event_candidate_count")
        ),
    }
    rows = []
    for invariant in REQUIRED_AUTOMATION_INVARIANTS:
        rows.append(
            {
                **invariant,
                "passed": bool(values.get(invariant["id"])),
            }
        )
    return rows


def _write_doc(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Automation Reliability Dossier",
        "",
        CLAIM_BOUNDARY,
        "",
        "## Summary",
        "",
        f"- Status: `{payload['status']}`",
        f"- Checks: `{payload['passed_count']}/{payload['check_count']}`",
        f"- External delivery enabled by default: `{payload['external_delivery_enabled_by_default']}`",
        f"- Real emergency coverage claim: `{payload['real_emergency_coverage_claim']}`",
        "",
        "## Required Invariants",
        "",
    ]
    for check in payload["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- `{mark}` `{check['id']}`: {check['description']}")
    lines.extend(["", "## Channel Matrix", ""])
    for channel in payload["channel_matrix"]:
        lines.append(f"- `{channel['channel']}`: {channel['intended_use']} Status: `{channel['live_patient_status']}`.")
    lines.extend(["", "## Automation Center Visibility Requirements", ""])
    for item in payload["automation_center_visibility_requirements"]:
        lines.append(f"- `{item['id']}`: {item['label']} - {item['why']}")
    lines.extend(
        [
            "",
            "## What This Does Not Prove",
            "",
            "- No clinical validation.",
            "- No clinician receipt or response guarantee.",
            "- No emergency coverage.",
            "- No compliance certification.",
            "- No healthcare production readiness.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_json(path: str) -> dict[str, Any]:
    target = _resolve(path)
    if not target.exists():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


__all__ = [
    "AUTOMATION_CENTER_VISIBILITY_REQUIREMENTS",
    "CHANNEL_MATRIX",
    "CLAIM_BOUNDARY",
    "DEFAULT_DOC_PATH",
    "DEFAULT_OUTPUT_PATH",
    "REQUIRED_AUTOMATION_INVARIANTS",
    "build_automation_reliability_dossier",
]
