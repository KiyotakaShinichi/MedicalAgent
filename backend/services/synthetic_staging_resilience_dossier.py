"""Consolidate executable local resilience drills without upgrading their claims."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_PATH = Path(
    "Data/evals/ops/latest_synthetic_staging_resilience_dossier.json"
)
SOURCES = {
    "automation_fault_injection": Path(
        "Data/evals/ops/latest_automation_fault_injection.json"
    ),
    "automation_channel_drill": Path(
        "Data/evals/ops/latest_automation_channel_drill.json"
    ),
    "deployment_recovery_drill": Path(
        "Data/evals/ops/latest_deployment_recovery_drill.json"
    ),
    "data_platform_reliability": Path(
        "Data/evals/ops/latest_data_platform_reliability_eval.json"
    ),
    "container_recovery_smoke": Path(
        "Data/evals/ops/latest_container_recovery_smoke.json"
    ),
    "managed_vector_shadow_sync": Path(
        "Data/evals/rag/latest_managed_vector_shadow_sync.json"
    ),
    "managed_vector_shadow_comparison": Path(
        "Data/evals/rag/latest_managed_vector_shadow_comparison.json"
    ),
    "automation_staging_readiness": Path(
        "Data/evals/ops/latest_synthetic_automation_staging_readiness.json"
    ),
    "disposable_staging_readiness": Path(
        "Data/evals/ops/latest_disposable_synthetic_staging_readiness.json"
    ),
}

CLAIM_BOUNDARY = (
    "This dossier consolidates disposable local and offline engineering drills "
    "over synthetic or curated non-patient fixtures. It does not prove managed-"
    "cloud durability, external channel reliability, clinician acknowledgement, "
    "emergency coverage, clinical validation, or production healthcare readiness."
)


def build_synthetic_staging_resilience_dossier(
    sources: dict[str, Path] | None = None,
) -> dict[str, Any]:
    paths = sources or SOURCES
    artifacts = {name: _read_json(path) for name, path in paths.items()}
    checks = [
        _check(
            "outbox_retry_dead_letter_requeue",
            artifacts["automation_fault_injection"].get("status") == "strong"
            and artifacts["automation_fault_injection"].get("pass_rate", 1.0) == 1.0,
            "local",
        ),
        _check(
            "signing_key_rotation_tamper_replay",
            _fault_scenario_passed(
                artifacts["automation_fault_injection"],
                "signing_key_rotation_and_tamper_rejection",
            )
            and _fault_scenario_passed(
                artifacts["automation_fault_injection"],
                "delayed_stale_event_rejected",
            ),
            "local",
        ),
        _check(
            "loopback_signed_delivery",
            artifacts["automation_channel_drill"].get("status") == "strong"
            and artifacts["automation_channel_drill"].get(
                "external_delivery_performed"
            )
            is False,
            "local",
        ),
        _check(
            "sqlite_backup_restore",
            artifacts["deployment_recovery_drill"].get("passed") is True
            and artifacts["deployment_recovery_drill"].get("postgres_restore_tested")
            is False,
            "local",
        ),
        _check(
            "data_replay_quarantine_tombstone_fallback",
            artifacts["data_platform_reliability"].get("failed") == 0
            and artifacts["data_platform_reliability"].get("n_cases", 0) >= 7,
            "local",
        ),
        _check(
            "compose_static_validation",
            artifacts["automation_staging_readiness"].get("status")
            == "ready_for_synthetic_runtime",
            "static_only",
        ),
    ]
    if "disposable_staging_readiness" in artifacts:
        checks.append(
            _check(
                "unified_disposable_staging_static_validation",
                artifacts["disposable_staging_readiness"].get("status")
                == "ready_for_disposable_synthetic_runtime"
                and artifacts["disposable_staging_readiness"].get(
                    "runtime_started"
                )
                is False,
                "static_only",
            )
        )
    local_passed = sum(check["passed"] for check in checks)
    blockers = [
        {
            "id": "container_runtime_drill",
            "resolved": artifacts["container_recovery_smoke"].get("completed")
            is True,
            "current_status": artifacts["container_recovery_smoke"].get("status"),
        },
        {
            "id": "managed_postgres_restore",
            "resolved": artifacts["deployment_recovery_drill"].get(
                "postgres_restore_tested"
            )
            is True,
            "current_status": "not_tested",
        },
        {
            "id": "managed_vector_sync_and_comparison",
            "resolved": artifacts["managed_vector_shadow_sync"].get("sync_completed")
            is True
            and artifacts["managed_vector_shadow_comparison"].get(
                "comparison_completed"
            )
            is True,
            "current_status": artifacts["managed_vector_shadow_comparison"].get(
                "status"
            ),
        },
        {
            "id": "azure_restore_and_failover",
            "resolved": artifacts["data_platform_reliability"]
            .get("recovery", {})
            .get("azure_restore_drill_completed")
            is True,
            "current_status": "not_tested",
        },
        {
            "id": "external_delivery_and_human_acknowledgement",
            "resolved": artifacts["automation_channel_drill"].get(
                "human_acknowledgement_proven"
            )
            is True,
            "current_status": "intentionally_not_claimed",
        },
    ]
    unresolved = [row for row in blockers if not row["resolved"]]
    return {
        "schema_version": "synthetic_staging_resilience_dossier_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "strong_local_only_external_blocked"
            if local_passed == len(checks)
            else "local_drill_needs_attention"
        ),
        "local_checks": checks,
        "local_passed_count": local_passed,
        "local_check_count": len(checks),
        "external_or_managed_blockers": blockers,
        "unresolved_external_or_managed_blocker_count": len(unresolved),
        "source_artifacts": {
            name: {
                "path": str(paths[name]).replace("\\", "/"),
                "status": artifact.get("status"),
                "generated_at": artifact.get("generated_at"),
            }
            for name, artifact in artifacts.items()
        },
        "staging_runtime_completed": False,
        "managed_cloud_drill_completed": False,
        "real_external_delivery_completed": False,
        "human_acknowledgement_proven": False,
        "patient_data_processed": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def write_synthetic_staging_resilience_dossier(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    payload = build_synthetic_staging_resilience_dossier()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _fault_scenario_passed(payload: dict[str, Any], scenario_id: str) -> bool:
    return any(
        row.get("id") == scenario_id and row.get("passed") is True
        for row in payload.get("scenarios") or []
        if isinstance(row, dict)
    )


def _check(check_id: str, passed: bool, evidence_scope: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "passed": bool(passed),
        "evidence_scope": evidence_scope,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


__all__ = [
    "build_synthetic_staging_resilience_dossier",
    "write_synthetic_staging_resilience_dossier",
]
