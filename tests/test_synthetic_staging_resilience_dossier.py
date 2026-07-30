import json
from pathlib import Path

from backend.services.synthetic_staging_resilience_dossier import (
    build_synthetic_staging_resilience_dossier,
)


def _write(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_local_strength_does_not_become_cloud_claim(tmp_path):
    sources = {
        "automation_fault_injection": _write(
            tmp_path / "fault.json",
            {
                "status": "strong",
                "pass_rate": 1.0,
                "scenarios": [
                    {"id": "signing_key_rotation_and_tamper_rejection", "passed": True},
                    {"id": "delayed_stale_event_rejected", "passed": True},
                ],
            },
        ),
        "automation_channel_drill": _write(
            tmp_path / "channel.json",
            {
                "status": "strong",
                "external_delivery_performed": False,
                "human_acknowledgement_proven": False,
            },
        ),
        "deployment_recovery_drill": _write(
            tmp_path / "restore.json",
            {"passed": True, "postgres_restore_tested": False},
        ),
        "data_platform_reliability": _write(
            tmp_path / "data.json",
            {
                "failed": 0,
                "n_cases": 7,
                "recovery": {"azure_restore_drill_completed": False},
            },
        ),
        "container_recovery_smoke": _write(
            tmp_path / "container.json",
            {"status": "blocked_environment", "completed": False},
        ),
        "managed_vector_shadow_sync": _write(
            tmp_path / "sync.json", {"sync_completed": False}
        ),
        "managed_vector_shadow_comparison": _write(
            tmp_path / "compare.json",
            {"status": "ready_for_managed_shadow_run", "comparison_completed": False},
        ),
        "automation_staging_readiness": _write(
            tmp_path / "staging.json", {"status": "ready_for_synthetic_runtime"}
        ),
    }
    result = build_synthetic_staging_resilience_dossier(sources)
    assert result["status"] == "strong_local_only_external_blocked"
    assert result["managed_cloud_drill_completed"] is False
    assert result["healthcare_production_ready"] is False
    assert result["clinical_validation"] is False
    assert result["unresolved_external_or_managed_blocker_count"] == 5
