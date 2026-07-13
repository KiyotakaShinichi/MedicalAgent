from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from backend.services.background_eval_worker import enqueue_job
from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_OUTPUT_PATH = "Data/evals/ops/latest_automation_control_plane.json"
DEFAULT_SCHEDULE_PATH = "Data/evals/ops/latest_automation_schedule_plan.json"
DEFAULT_DOC_PATH = "docs/automation_control_plane.md"

CLAIM_BOUNDARY = (
    "The automation control plane schedules and summarizes redacted engineering work only. It must not send PHI, "
    "message patients, issue medical guidance, trigger unreviewed clinical escalation, or weaken safety gates. "
    "It is not clinical validation, compliance certification, or healthcare production readiness."
)

SCHEDULES: tuple[dict[str, Any], ...] = (
    {
        "id": "nightly_core_eval_refresh",
        "cadence": "daily",
        "interval_hours": 24,
        "jobs": [
            "refresh_trace_envelope_v2_eval",
            "refresh_runtime_quality_sentinel",
            "refresh_eval_history",
            "refresh_release_gate_explanation",
            "run_release_gate",
        ],
    },
    {
        "id": "weekly_integration_shadow_refresh",
        "cadence": "weekly",
        "interval_hours": 168,
        "jobs": [
            "refresh_pinecone_shadow_retrieval",
            "refresh_n8n_templates",
            "refresh_external_dataset_matrix",
            "refresh_platform_control_plane",
        ],
    },
    {
        "id": "weekly_security_health_refresh",
        "cadence": "weekly",
        "interval_hours": 168,
        "jobs": ["refresh_dependency_security_scan", "refresh_ops_health_snapshot"],
    },
    {
        "id": "biweekly_reviewer_reminder",
        "cadence": "every_14_days",
        "interval_hours": 336,
        "jobs": ["prepare_reviewer_packet_reminder"],
    },
)


def build_automation_schedule_plan(
    *,
    output_path: str | Path = DEFAULT_SCHEDULE_PATH,
) -> dict[str, Any]:
    payload = {
        "schema_version": "automation_schedule_plan_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_scheduler_or_n8n",
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "phi_allowed": False,
        "scheduler_installed": False,
        "schedules": list(SCHEDULES),
        "event_driven_jobs": [
            "publish_release_gate_alert",
            "create_stale_artifact_ticket",
            "publish_trace_quality_digest",
            "publish_pinecone_shadow_report",
            "publish_dependency_security_alert",
            "publish_deployment_health_alert",
            "publish_external_red_team_intake",
        ],
        "deployment_options": [
            "n8n Schedule Trigger calls the admin automation API with a bearer token stored in n8n credentials.",
            "A CI schedule runs scripts/run_automation_control_plane.py and approved refresh scripts.",
            "A host cron or Windows Task Scheduler invokes the preview/worker scripts outside the API process.",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(_resolve(output_path), payload)
    return payload


def due_schedule_jobs(
    *,
    now: datetime,
    last_run_by_schedule: Mapping[str, datetime | str | None],
) -> list[dict[str, Any]]:
    due: list[dict[str, Any]] = []
    for schedule in SCHEDULES:
        last_run = _as_datetime(last_run_by_schedule.get(schedule["id"]))
        interval = timedelta(hours=int(schedule["interval_hours"]))
        if last_run is None or now - last_run >= interval:
            for job_type in schedule["jobs"]:
                due.append(
                    {
                        "schedule_id": schedule["id"],
                        "job_type": job_type,
                        "dry_run": True,
                        "reason": f"scheduled_{schedule['cadence']}_refresh",
                    }
                )
    return due


def build_automation_control_plane(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    schedule_path: str | Path = DEFAULT_SCHEDULE_PATH,
    doc_path: str | Path = DEFAULT_DOC_PATH,
) -> dict[str, Any]:
    schedule = build_automation_schedule_plan(output_path=schedule_path)
    events = _engineering_event_jobs()
    jobs = [
        enqueue_job(
            job_type=event["job_type"],
            requested_by="automation_control_plane",
            payload=event["payload"],
            dry_run=True,
        )
        for event in events
    ]
    payload = {
        "schema_version": "automation_control_plane_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if all(job["accepted"] for job in jobs) else "needs_attention",
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "phi_allowed": False,
        "live_patient_route_enabled": False,
        "commands_executed": False,
        "webhooks_sent": False,
        "schedule_status": schedule["status"],
        "schedule_count": len(schedule["schedules"]),
        "event_candidate_count": len(events),
        "accepted_event_count": sum(1 for job in jobs if job["accepted"]),
        "rejected_event_count": sum(1 for job in jobs if not job["accepted"]),
        "event_jobs": jobs,
        "source_artifacts": {
            "release_gate": "Data/evals/governance/latest_release_gate_explanation.json",
            "trace_envelope": "Data/evals/ops/latest_trace_envelope_v2_eval.json",
            "pinecone_shadow": "Data/evals/rag/latest_pinecone_shadow_retrieval_comparison.json",
            "dependency_security": "Data/evals/ops/latest_dependency_security_scan.json",
            "service_health": "Data/evals/ops/latest_service_health_snapshot.json",
            "external_review": "Data/evals/governance/latest_external_review_execution_readiness.json",
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(_resolve(output_path), payload)
    _write_doc(_resolve(doc_path), payload, schedule)
    return payload


def _engineering_event_jobs() -> list[dict[str, Any]]:
    release_gate = _read_json("Data/evals/governance/latest_release_gate_explanation.json")
    trace = _read_json("Data/evals/ops/latest_trace_envelope_v2_eval.json")
    pinecone = _read_json("Data/evals/rag/latest_pinecone_shadow_retrieval_comparison.json")
    dependency = _read_json("Data/evals/ops/latest_dependency_security_scan.json")
    health = _read_json("Data/evals/ops/latest_service_health_snapshot.json")
    review = _read_json("Data/evals/governance/latest_external_review_execution_readiness.json")

    health_metrics = health.get("metrics") or {}
    dependency_summary = dependency.get("summary") or {}
    local = (pinecone.get("local_reference_metrics") or {}).get("source_governed_full_stack") or {}
    delta = pinecone.get("delta_vs_local") or {}
    now_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pending_reviews = list(review.get("pending_review_types") or [])
    next_review = pending_reviews[0] if pending_reviews else "external_peer_review"

    events = [
        {
            "job_type": "publish_release_gate_alert",
            "payload": {
                "run_id": now_id,
                "status": release_gate.get("status", "unknown"),
                "artifact_count": None,
                "failure_count": 0 if release_gate.get("status") == "strong" else None,
                "summary_url": "/artifacts/evals/governance/latest_release_gate_explanation.json",
            },
        },
        {
            "job_type": "publish_trace_quality_digest",
            "payload": {
                "run_id": now_id,
                "trace_coverage_rate": trace.get("validation_pass_rate"),
                "missing_fields_count": 0 if trace.get("validation_pass_rate") == 1.0 else None,
                "dashboard_path": "/artifacts/evals/ops/latest_trace_envelope_v2_eval.json",
            },
        },
        {
            "job_type": "publish_pinecone_shadow_report",
            "payload": {
                "run_id": now_id,
                "status": pinecone.get("status", "unknown"),
                "recall_at_10_delta": delta.get("recall_at_10"),
                "citation_precision_delta": delta.get("citation_precision"),
                "source_tier_correctness": local.get("source_tier_correctness"),
                "promotion_allowed": bool((pinecone.get("promotion_gate") or {}).get("pinecone_can_replace_local_retrieval")),
            },
        },
        {
            "job_type": "publish_dependency_security_alert",
            "payload": {
                "run_id": now_id,
                "status": dependency.get("status", "unknown"),
                "finding_count": dependency_summary.get("high_or_critical_count"),
                "tools_available": max(
                    0,
                    int(dependency_summary.get("tool_count") or 0)
                    - int(dependency_summary.get("unavailable_tool_count") or 0),
                ),
                "artifact_path": "Data/evals/ops/latest_dependency_security_scan.json",
            },
        },
        {
            "job_type": "publish_deployment_health_alert",
            "payload": {
                "run_id": now_id,
                "status": health.get("status", "unknown"),
                "failed_benchmark_count": health_metrics.get("failed_benchmark_count"),
                "stale_artifact_count": health_metrics.get("stale_artifact_count"),
                "artifact_path": "Data/evals/ops/latest_service_health_snapshot.json",
            },
        },
        {
            "job_type": "prepare_reviewer_packet_reminder",
            "payload": {
                "review_type": next_review,
                "packet_path": "docs/review_packets/",
                "attestation_template_path": "Data/evals/external_review/reviewer_attestation_template.md",
                "due_date": "reviewer_selected_date",
            },
        },
    ]
    stale = list(health.get("stale_artifacts") or [])
    if stale:
        events.append(
            {
                "job_type": "create_stale_artifact_ticket",
                "payload": {
                    "run_id": now_id,
                    "artifact_path": str(stale[0]),
                    "age_days": None,
                    "severity": "warning",
                    "owner": "engineering",
                },
            }
        )
    return events


def _write_doc(path: Path, payload: Mapping[str, Any], schedule: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Automation Control Plane",
        "",
        CLAIM_BOUNDARY,
        "",
        "## Current State",
        "",
        f"- Status: `{payload['status']}`",
        f"- Commands executed while building artifact: `{payload['commands_executed']}`",
        f"- Webhooks sent while building artifact: `{payload['webhooks_sent']}`",
        f"- Event candidates: `{payload['event_candidate_count']}`",
        "",
        "## Schedules",
        "",
    ]
    for item in schedule["schedules"]:
        lines.append(f"- `{item['id']}` ({item['cadence']}): {', '.join(item['jobs'])}")
    lines.extend(
        [
            "",
            "## Runtime Contract",
            "",
            "- API access is admin-only.",
            "- Jobs default to dry-run.",
            "- Local execution requires `NLCARE_AUTOMATION_EXECUTION_ENABLED=true`.",
            "- n8n dispatch requires a configured URL, signing secret, and explicit enable flag.",
            "- Scheduled installation remains an operator/deployment choice; no host scheduler is installed automatically.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _as_datetime(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


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
    "CLAIM_BOUNDARY",
    "SCHEDULES",
    "build_automation_control_plane",
    "build_automation_schedule_plan",
    "due_schedule_jobs",
]
