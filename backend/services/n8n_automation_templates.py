from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_OUTPUT_PATH = "Data/evals/ops/latest_n8n_workflow_templates.json"
DEFAULT_DOC_PATH = "docs/n8n_internal_automation_templates.md"
DEFAULT_TEMPLATE_DIR = "Data/evals/ops/n8n_workflow_templates"

CLAIM_BOUNDARY = (
    "n8n workflow templates are internal automation scaffolds only. They may notify maintainers, create admin "
    "tickets, remind reviewers, or trigger internal eval refreshes. They must not send PHI, issue medical advice, "
    "override safety validators, or automate diagnosis, treatment, medication, prognosis, genetics, tumor-marker, "
    "or clinical-escalation decisions."
)

BLOCKED_PAYLOAD_FIELDS = [
    "patient_name",
    "patient_id",
    "raw_patient_message",
    "full_chat_transcript",
    "medical_record_number",
    "date_of_birth",
    "address",
    "phone",
    "email",
    "raw_prompt",
    "raw_response",
    "raw_trace",
    "private_chain_of_thought",
    "genetic_variant_details_for_patient_advice",
]


def build_n8n_automation_templates(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    doc_path: str | Path = DEFAULT_DOC_PATH,
    template_dir: str | Path = DEFAULT_TEMPLATE_DIR,
) -> dict[str, Any]:
    templates = _templates()
    template_root = _resolve(template_dir)
    template_root.mkdir(parents=True, exist_ok=True)
    written = []
    for template in templates:
        path = template_root / f"{template['id']}.json"
        path.write_text(json.dumps(template["workflow"], indent=2), encoding="utf-8")
        written.append(str(_relative_to_root(path)))

    payload = {
        "schema_version": "n8n_workflow_templates_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_optional_import",
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "phi_allowed": False,
        "live_patient_route_enabled": False,
        "template_count": len(templates),
        "templates": [
            {
                "id": item["id"],
                "title": item["title"],
                "allowed_use": item["allowed_use"],
                "not_allowed_use": item["not_allowed_use"],
                "template_path": written[idx],
            }
            for idx, item in enumerate(templates)
        ],
        "import_instructions": [
            "Import templates manually into n8n; do not commit credentials.",
            "Replace placeholder webhook URLs and notification channels in n8n UI.",
            "Use test webhook URLs first.",
            "Use signed payloads from NLCare when connecting FastAPI to n8n.",
            "Keep workflows admin-only until compliance/security review exists.",
        ],
        "blocked_payload_fields": BLOCKED_PAYLOAD_FIELDS,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(_resolve(output_path), payload)
    _write_doc(_resolve(doc_path), payload)
    return payload


def _templates() -> list[dict[str, Any]]:
    return [
        {
            "id": "release_gate_alert",
            "title": "Release Gate Alert",
            "allowed_use": "Notify maintainers when release gate passes or fails.",
            "not_allowed_use": "Do not expose patient data or clinical conclusions.",
            "workflow": _workflow(
                name="NLCare Release Gate Alert",
                trigger_path="nlcare/release-gate",
                action_summary="Post release-gate status to an internal notification channel.",
                required_fields=["run_id", "status", "artifact_count", "failure_count", "summary_url"],
            ),
        },
        {
            "id": "stale_artifact_ticket",
            "title": "Stale Artifact Ticket",
            "allowed_use": "Create an admin task when critical eval artifacts are stale.",
            "not_allowed_use": "Do not downgrade safety thresholds or auto-refresh patient-facing outputs.",
            "workflow": _workflow(
                name="NLCare Stale Artifact Ticket",
                trigger_path="nlcare/stale-artifact",
                action_summary="Create an internal ticket for stale blocker/warning artifacts.",
                required_fields=["run_id", "artifact_path", "age_days", "severity", "owner"],
            ),
        },
        {
            "id": "reviewer_intake_reminder",
            "title": "Reviewer Intake Reminder",
            "allowed_use": "Send reminder/checklist for external reviewer packets.",
            "not_allowed_use": "Do not imply clinician approval or clinical validation.",
            "workflow": _workflow(
                name="NLCare Reviewer Intake Reminder",
                trigger_path="nlcare/reviewer-intake",
                action_summary="Send reviewer packet link and attestation checklist.",
                required_fields=["review_type", "packet_path", "attestation_template_path", "due_date"],
            ),
        },
        {
            "id": "eval_refresh_trigger",
            "title": "Eval Refresh Trigger",
            "allowed_use": "Trigger internal eval refresh jobs for synthetic/non-live artifacts.",
            "not_allowed_use": "Do not trigger clinical actions or use raw patient chat payloads.",
            "workflow": _workflow(
                name="NLCare Eval Refresh Trigger",
                trigger_path="nlcare/eval-refresh",
                action_summary="Call an internal admin endpoint to refresh a named non-live eval.",
                required_fields=["run_id", "eval_name", "requested_by", "reason"],
            ),
        },
        {
            "id": "trace_quality_digest",
            "title": "Trace Quality Digest",
            "allowed_use": "Notify maintainers when trace coverage or trace-envelope validation needs attention.",
            "not_allowed_use": "Do not send raw prompts, raw responses, private chain-of-thought, or PHI.",
            "workflow": _workflow(
                name="NLCare Trace Quality Digest",
                trigger_path="nlcare/trace-quality",
                action_summary="Post redacted trace-quality status to an internal engineering channel.",
                required_fields=["run_id", "trace_coverage_rate", "missing_fields_count", "dashboard_path"],
            ),
        },
        {
            "id": "pinecone_shadow_report",
            "title": "Pinecone Shadow Report",
            "allowed_use": "Send managed-vector shadow retrieval metrics for engineering review.",
            "not_allowed_use": "Do not promote Pinecone to live retrieval or store patient chat content.",
            "workflow": _workflow(
                name="NLCare Pinecone Shadow Report",
                trigger_path="nlcare/pinecone-shadow",
                action_summary="Post redacted Pinecone shadow retrieval comparison metrics.",
                required_fields=[
                    "run_id",
                    "status",
                    "recall_at_10_delta",
                    "citation_precision_delta",
                    "source_tier_correctness",
                    "promotion_allowed",
                ],
            ),
        },
        {
            "id": "external_red_team_intake",
            "title": "External Red-Team Intake",
            "allowed_use": "Send no-read red-team packet links and attestation checklist to a reviewer.",
            "not_allowed_use": "Do not imply clinical validation, clinician approval, or completed external review.",
            "workflow": _workflow(
                name="NLCare External Red-Team Intake",
                trigger_path="nlcare/external-red-team-intake",
                action_summary="Send reviewer packet link and anti-contamination checklist.",
                required_fields=["review_type", "packet_path", "attestation_template_path", "due_date"],
            ),
        },
        {
            "id": "dependency_security_alert",
            "title": "Dependency Security Alert",
            "allowed_use": "Create an internal ticket for dependency/security-scan findings.",
            "not_allowed_use": "Do not label dependency scan pass/fail as HIPAA, SOC 2, or healthcare compliance.",
            "workflow": _workflow(
                name="NLCare Dependency Security Alert",
                trigger_path="nlcare/dependency-security",
                action_summary="Create an internal engineering ticket for dependency/security findings.",
                required_fields=["run_id", "severity", "package", "advisory_id", "remediation_owner"],
            ),
        },
        {
            "id": "deployment_health_alert",
            "title": "Deployment Health Alert",
            "allowed_use": "Notify maintainers about demo service health and stale engineering artifacts.",
            "not_allowed_use": "Do not present engineering health as clinical safety or send patient data.",
            "workflow": _workflow(
                name="NLCare Deployment Health Alert",
                trigger_path="nlcare/deployment-health",
                action_summary="Post redacted engineering health status to an internal operations channel.",
                required_fields=[
                    "run_id",
                    "status",
                    "failed_benchmark_count",
                    "stale_artifact_count",
                    "artifact_path",
                ],
            ),
        },
        {
            "id": "high_risk_review_alert",
            "title": "High-Priority Conversation Review Alert",
            "allowed_use": (
                "Notify an approved internal reviewer channel that a redacted NLCare review item is waiting. "
                "Operators may attach an email, SMS, or Viber node after access-control review."
            ),
            "not_allowed_use": (
                "Do not send patient identifiers, raw chat text, medical conclusions, or imply that delivery "
                "means a clinician saw or acted on the alert."
            ),
            "workflow": _workflow(
                name="NLCare High-Priority Conversation Review Alert",
                trigger_path="nlcare/high-risk-review-alert",
                action_summary=(
                    "Send a redacted internal notification with a sign-in-required review-item link."
                ),
                required_fields=[
                    "alert_id",
                    "event_type",
                    "priority",
                    "review_path",
                    "delivery_scope",
                    "recipient_scope",
                ],
                test_recipient_only=True,
                delivery_receipt_required=True,
            ),
        },
    ]


def _workflow(
    *,
    name: str,
    trigger_path: str,
    action_summary: str,
    required_fields: list[str],
    test_recipient_only: bool = False,
    delivery_receipt_required: bool = False,
) -> dict[str, Any]:
    # This intentionally uses generic nodes so the JSON stays importable-ish
    # without credentials. Teams should wire real notification/ticket nodes in n8n.
    return {
        "name": name,
        "active": False,
        "nodes": [
            {
                "parameters": {
                    "path": trigger_path,
                    "httpMethod": "POST",
                    "responseMode": "responseNode",
                    "options": {},
                },
                "id": "webhook-trigger",
                "name": "Webhook Trigger",
                "type": "n8n-nodes-base.webhook",
                "typeVersion": 2,
                "position": [0, 0],
            },
            {
                "parameters": {
                    "jsCode": (
                        "const event = $json.body || $json;\n"
                        "const body = event.payload || event;\n"
                        "const headers = $json.headers || {};\n"
                        "const signaturePresent = Boolean(headers['x-nlcare-signature'] || headers['X-NLCare-Signature']);\n"
                        "const timestampPresent = Boolean(headers['x-nlcare-timestamp'] || headers['X-NLCare-Timestamp']);\n"
                        f"const required = {json.dumps(required_fields)};\n"
                        "const missing = required.filter((key) => body[key] === undefined || body[key] === null || body[key] === '');\n"
                        f"const blocked = {json.dumps(BLOCKED_PAYLOAD_FIELDS)};\n"
                        "const blockedPresent = blocked.filter((key) => body[key] !== undefined);\n"
                        "const boundaryOk = event.phi_allowed === false && event.clinical_validation === false;\n"
                        "return [{ json: { ok: signaturePresent && timestampPresent && boundaryOk && missing.length === 0 && blockedPresent.length === 0, signaturePresent, timestampPresent, boundaryOk, missing, blockedPresent, body } }];"
                    )
                },
                "id": "validate-payload",
                "name": "Validate Redacted Payload",
                "type": "n8n-nodes-base.code",
                "typeVersion": 2,
                "position": [240, 0],
            },
            {
                "parameters": {
                    "respondWith": "json",
                    "responseBody": (
                        "={{ { ok: $json.ok, action: "
                        + json.dumps(action_summary)
                        + ", missing: $json.missing, blockedPresent: $json.blockedPresent } }}"
                    ),
                    "options": {},
                },
                "id": "respond",
                "name": "Return Workflow Status",
                "type": "n8n-nodes-base.respondToWebhook",
                "typeVersion": 1,
                "position": [480, 0],
            },
        ],
        "connections": {
            "Webhook Trigger": {"main": [[{"node": "Validate Redacted Payload", "type": "main", "index": 0}]]},
            "Validate Redacted Payload": {"main": [[{"node": "Return Workflow Status", "type": "main", "index": 0}]]},
        },
        "settings": {"executionOrder": "v1"},
        "staticData": None,
        "meta": {
            "template": True,
            "clinical_validation": False,
            "phi_allowed": False,
            "signature_header_presence_required": True,
            "timestamp_header_presence_required": True,
            "receiver_hmac_verification_requires_operator_configuration": True,
            "receiver_replay_window_seconds": 300,
            "test_recipient_only": test_recipient_only,
            "delivery_receipt_callback_required": delivery_receipt_required,
            "delivery_receipt_is_not_clinician_acknowledgement": True,
            "claim_boundary": CLAIM_BOUNDARY,
        },
    }


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# n8n Internal Automation Templates",
        "",
        payload["claim_boundary"],
        "",
        "## Templates",
        "",
    ]
    for item in payload["templates"]:
        lines.extend(
            [
                f"- **{item['title']}** (`{item['id']}`)",
                f"  - Path: `{item['template_path']}`",
                f"  - Allowed: {item['allowed_use']}",
                f"  - Not allowed: {item['not_allowed_use']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Import Instructions",
            "",
            *[f"- {item}" for item in payload["import_instructions"]],
            "",
            "## Blocked Payload Fields",
            "",
            *[f"- `{item}`" for item in payload["blocked_payload_fields"]],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


def _relative_to_root(path: Path) -> Path:
    try:
        return path.relative_to(ROOT_DIR)
    except ValueError:
        return path


__all__ = [
    "CLAIM_BOUNDARY",
    "DEFAULT_DOC_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_TEMPLATE_DIR",
    "build_n8n_automation_templates",
]
