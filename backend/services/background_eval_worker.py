from __future__ import annotations

import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_OUTPUT_PATH = "Data/evals/ops/latest_background_eval_worker_dry_run.json"
DEFAULT_DOC_PATH = "docs/background_eval_worker.md"

CLAIM_BOUNDARY = (
    "Background eval worker is an admin-only automation scaffold for redacted engineering jobs. "
    "It is not clinical validation, not healthcare production readiness, and cannot execute diagnosis, "
    "treatment, medication, prognosis, genetics, tumor-marker, or clinical-escalation actions."
)

ALLOWED_JOB_TYPES: dict[str, dict[str, Any]] = {
    "run_release_gate": {
        "mode": "local_command",
        "command": ["python", "scripts/run_release_gate.py"],
        "allowed_payload_fields": ["run_id", "requested_by", "reason"],
    },
    "refresh_trace_envelope_v2_eval": {
        "mode": "local_command",
        "command": ["python", "scripts/run_trace_envelope_v2_eval.py"],
        "allowed_payload_fields": ["run_id", "requested_by", "reason"],
    },
    "refresh_pinecone_shadow_retrieval": {
        "mode": "local_command",
        "command": ["python", "scripts/run_pinecone_shadow_retrieval.py"],
        "allowed_payload_fields": ["run_id", "requested_by", "reason"],
    },
    "refresh_n8n_templates": {
        "mode": "local_command",
        "command": ["python", "scripts/run_n8n_automation_templates.py"],
        "allowed_payload_fields": ["run_id", "requested_by", "reason"],
    },
    "refresh_external_dataset_matrix": {
        "mode": "local_command",
        "command": ["python", "scripts/run_external_dataset_integration_matrix.py"],
        "allowed_payload_fields": ["run_id", "requested_by", "reason"],
    },
    "refresh_platform_control_plane": {
        "mode": "local_command",
        "command": ["python", "scripts/run_platform_control_plane_architecture.py"],
        "allowed_payload_fields": ["run_id", "requested_by", "reason"],
    },
    "refresh_runtime_quality_sentinel": {
        "mode": "local_command",
        "command": ["python", "scripts/run_runtime_quality_sentinel.py"],
        "allowed_payload_fields": ["run_id", "requested_by", "reason"],
    },
    "refresh_eval_history": {
        "mode": "local_command",
        "command": ["python", "scripts/update_eval_history.py"],
        "allowed_payload_fields": ["run_id", "requested_by", "reason"],
    },
    "refresh_dependency_security_scan": {
        "mode": "local_command",
        "command": ["python", "scripts/run_dependency_security_scan.py"],
        "allowed_payload_fields": ["run_id", "requested_by", "reason"],
    },
    "refresh_ops_health_snapshot": {
        "mode": "local_command",
        "command": ["python", "scripts/run_ops_health_snapshot.py"],
        "allowed_payload_fields": ["run_id", "requested_by", "reason"],
    },
    "refresh_release_gate_explanation": {
        "mode": "local_command",
        "command": ["python", "scripts/run_release_gate_explanation.py"],
        "allowed_payload_fields": ["run_id", "requested_by", "reason"],
    },
    "create_stale_artifact_ticket": {
        "mode": "webhook_payload_only",
        "command": None,
        "allowed_payload_fields": ["run_id", "artifact_path", "age_days", "severity", "owner"],
    },
    "prepare_reviewer_packet_reminder": {
        "mode": "webhook_payload_only",
        "command": None,
        "allowed_payload_fields": ["review_type", "packet_path", "attestation_template_path", "due_date"],
    },
    "publish_trace_quality_digest": {
        "mode": "webhook_payload_only",
        "command": None,
        "allowed_payload_fields": ["run_id", "trace_coverage_rate", "missing_fields_count", "dashboard_path"],
    },
    "publish_pinecone_shadow_report": {
        "mode": "webhook_payload_only",
        "command": None,
        "workflow_id": "pinecone_shadow_report",
        "allowed_payload_fields": [
            "run_id",
            "status",
            "recall_at_10_delta",
            "citation_precision_delta",
            "source_tier_correctness",
            "promotion_allowed",
        ],
    },
    "publish_external_red_team_intake": {
        "mode": "webhook_payload_only",
        "command": None,
        "workflow_id": "external_red_team_intake",
        "allowed_payload_fields": ["submission_id", "author_role", "case_count", "attestation_present", "artifact_path"],
    },
    "publish_dependency_security_alert": {
        "mode": "webhook_payload_only",
        "command": None,
        "workflow_id": "dependency_security_alert",
        "allowed_payload_fields": ["run_id", "status", "finding_count", "tools_available", "artifact_path"],
    },
    "publish_deployment_health_alert": {
        "mode": "webhook_payload_only",
        "command": None,
        "workflow_id": "deployment_health_alert",
        "allowed_payload_fields": ["run_id", "status", "failed_benchmark_count", "stale_artifact_count", "artifact_path"],
    },
    "publish_release_gate_alert": {
        "mode": "webhook_payload_only",
        "command": None,
        "workflow_id": "release_gate_alert",
        "allowed_payload_fields": ["run_id", "status", "artifact_count", "failure_count", "summary_url"],
    },
}

BLOCKED_JOB_TYPES: frozenset[str] = frozenset(
    {
        "diagnosis",
        "treatment_recommendation",
        "dosage_change",
        "prognosis",
        "genetic_risk_interpretation",
        "tumor_marker_interpretation",
        "clinical_escalation_without_human_review",
        "send_phi_to_external_service",
        "message_patient_directly",
    }
)

BLOCKED_PAYLOAD_FIELDS: frozenset[str] = frozenset(
    {
        "patient_name",
        "patient_id",
        "medical_record_number",
        "date_of_birth",
        "address",
        "phone",
        "email",
        "raw_patient_message",
        "full_chat_transcript",
        "raw_prompt",
        "raw_response",
        "private_chain_of_thought",
        "genetic_variant_details_for_patient_advice",
    }
)


def enqueue_job(
    *,
    job_type: str,
    requested_by: str,
    payload: Mapping[str, Any] | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    payload = dict(payload or {})
    blocked_payload_fields = _find_blocked_payload_fields(payload)
    allowed = ALLOWED_JOB_TYPES.get(job_type)
    rejected_reason = None
    if job_type in BLOCKED_JOB_TYPES:
        rejected_reason = "blocked_clinical_or_phi_action"
    elif allowed is None:
        rejected_reason = "unknown_job_type"
    elif blocked_payload_fields:
        rejected_reason = "blocked_payload_fields_present"

    accepted = rejected_reason is None
    allowed_fields = set((allowed or {}).get("allowed_payload_fields") or [])
    sanitized_payload = {
        key: value
        for key, value in payload.items()
        if key in allowed_fields and key not in BLOCKED_PAYLOAD_FIELDS
    }
    return {
        "job_id": str(uuid.uuid4()),
        "job_type": job_type,
        "requested_by": requested_by,
        "accepted": accepted,
        "rejected_reason": rejected_reason,
        "dry_run": dry_run,
        "mode": (allowed or {}).get("mode"),
        "command": (allowed or {}).get("command") if accepted else None,
        "payload_redacted": True,
        "sanitized_payload": sanitized_payload,
        "blocked_payload_fields": blocked_payload_fields,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "clinical_validation": False,
    }


def execute_job(
    job: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    if not job.get("accepted"):
        raise ValueError(job.get("rejected_reason") or "automation_job_rejected")
    job_type = str(job.get("job_type") or "")
    definition = ALLOWED_JOB_TYPES.get(job_type)
    if definition is None:
        raise ValueError("unknown_job_type")
    payload = dict(job.get("sanitized_payload") or {})
    blocked = _find_blocked_payload_fields(payload)
    if blocked:
        raise ValueError(f"blocked_payload_fields_present: {blocked}")

    if definition["mode"] == "webhook_payload_only":
        from backend.services.n8n_webhook_dispatcher import dispatch_signed_webhook

        workflow_id = definition.get("workflow_id") or _workflow_id_for_job(job_type)
        return dispatch_signed_webhook(
            workflow_id=workflow_id,
            payload=payload,
            env=env,
        )

    if job.get("dry_run", True):
        return {
            "status": "dry_run_completed",
            "job_type": job_type,
            "command_preview": list(definition.get("command") or []),
            "commands_executed": False,
            "clinical_validation": False,
            "claim_boundary": CLAIM_BOUNDARY,
        }

    values = dict(os.environ if env is None else env)
    if not _truthy(values.get("NLCARE_AUTOMATION_EXECUTION_ENABLED")):
        raise PermissionError("Automation execution is disabled; set NLCARE_AUTOMATION_EXECUTION_ENABLED=true")
    command = list(definition.get("command") or [])
    if not command or command[0] not in {"python", "python.exe"}:
        raise ValueError("invalid_allowlisted_command")
    completed = subprocess.run(
        command,
        cwd=ROOT_DIR,
        env=values,
        capture_output=True,
        text=True,
        timeout=max(1, min(int(timeout_seconds), 1800)),
        shell=False,
        check=False,
    )
    output = _bounded_output(completed.stdout, completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"Automation command failed with exit_code={completed.returncode}: {output}")
    return {
        "status": "completed",
        "job_type": job_type,
        "exit_code": completed.returncode,
        "commands_executed": True,
        "output_preview": output,
        "clinical_validation": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def build_background_eval_worker_dry_run(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    doc_path: str | Path = DEFAULT_DOC_PATH,
) -> dict[str, Any]:
    sample_jobs = [
        enqueue_job(
            job_type="run_release_gate",
            requested_by="admin",
            payload={"run_id": "dry-run-001", "requested_by": "admin", "reason": "manual check"},
        ),
        enqueue_job(
            job_type="refresh_trace_envelope_v2_eval",
            requested_by="admin",
            payload={"run_id": "dry-run-002", "requested_by": "admin", "reason": "trace contract refresh"},
        ),
        enqueue_job(
            job_type="create_stale_artifact_ticket",
            requested_by="system",
            payload={
                "run_id": "dry-run-003",
                "artifact_path": "Data/evals/safety/latest_adversarial_eval.json",
                "age_days": 39,
                "severity": "warning",
                "owner": "engineering",
            },
        ),
        enqueue_job(
            job_type="diagnosis",
            requested_by="malicious_test",
            payload={"patient_id": "P001", "raw_patient_message": "Do I have recurrence?"},
        ),
    ]
    accepted_count = sum(1 for job in sample_jobs if job["accepted"])
    rejected_count = len(sample_jobs) - accepted_count
    payload = {
        "schema_version": "background_eval_worker_dry_run_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong",
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "phi_allowed": False,
        "live_patient_route_enabled": False,
        "commands_executed": False,
        "accepted_job_count": accepted_count,
        "rejected_job_count": rejected_count,
        "allowed_job_types": sorted(ALLOWED_JOB_TYPES),
        "blocked_job_types": sorted(BLOCKED_JOB_TYPES),
        "blocked_payload_fields": sorted(BLOCKED_PAYLOAD_FIELDS),
        "sample_jobs": sample_jobs,
        "automation_upgrade_path": [
            "Use the admin-only /admin/automation API to enqueue redacted jobs.",
            "Keep local execution behind NLCARE_AUTOMATION_EXECUTION_ENABLED and bounded subprocess timeouts.",
            "Use HMAC-signed n8n webhooks for internal notifications only.",
            "Store job outputs as governance artifacts, never as patient-facing medical advice.",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(_resolve(output_path), payload)
    _write_doc(_resolve(doc_path), payload)
    return payload


def _write_doc(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Background Eval Worker",
        "",
        CLAIM_BOUNDARY,
        "",
        "## Dry-Run Status",
        "",
        f"- Status: `{payload['status']}`",
        f"- Commands executed: `{payload['commands_executed']}`",
        f"- Accepted jobs: `{payload['accepted_job_count']}`",
        f"- Rejected jobs: `{payload['rejected_job_count']}`",
        "",
        "## Allowed Job Types",
        "",
        *[f"- `{job}`" for job in payload["allowed_job_types"]],
        "",
        "## Blocked Job Types",
        "",
        *[f"- `{job}`" for job in payload["blocked_job_types"]],
        "",
        "## Blocked Payload Fields",
        "",
        *[f"- `{field}`" for field in payload["blocked_payload_fields"]],
        "",
        "## Automation Upgrade Path",
        "",
        *[f"- {step}" for step in payload["automation_upgrade_path"]],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


def _find_blocked_payload_fields(payload: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            if key.lower() in BLOCKED_PAYLOAD_FIELDS:
                found.append(path)
            found.extend(_find_blocked_payload_fields(value, path))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            found.extend(_find_blocked_payload_fields(value, path))
    return sorted(set(found))


def _workflow_id_for_job(job_type: str) -> str:
    mapping = {
        "create_stale_artifact_ticket": "stale_artifact_ticket",
        "prepare_reviewer_packet_reminder": "reviewer_intake_reminder",
        "publish_trace_quality_digest": "trace_quality_digest",
    }
    workflow_id = mapping.get(job_type)
    if workflow_id is None:
        raise ValueError(f"No n8n workflow mapped for job_type={job_type}")
    return workflow_id


def _bounded_output(stdout: str | None, stderr: str | None) -> str:
    combined = "\n".join(part.strip() for part in (stdout or "", stderr or "") if part.strip())
    return combined[-4000:]


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


__all__ = [
    "ALLOWED_JOB_TYPES",
    "BLOCKED_JOB_TYPES",
    "BLOCKED_PAYLOAD_FIELDS",
    "CLAIM_BOUNDARY",
    "DEFAULT_DOC_PATH",
    "DEFAULT_OUTPUT_PATH",
    "build_background_eval_worker_dry_run",
    "execute_job",
    "enqueue_job",
]
