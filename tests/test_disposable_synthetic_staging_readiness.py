from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from backend.services.disposable_synthetic_staging_readiness import (
    REQUIRED_SERVICES,
    build_disposable_synthetic_staging_readiness,
    write_disposable_synthetic_staging_readiness,
)


def test_disposable_staging_is_loopback_synthetic_and_fail_closed():
    payload = build_disposable_synthetic_staging_readiness()
    assert set(payload["services"]) == REQUIRED_SERVICES
    assert payload["passed_count"] == payload["check_count"]
    assert payload["status"] == "ready_for_disposable_synthetic_runtime"
    assert payload["runtime_started"] is False
    assert payload["real_external_delivery_completed"] is False
    assert payload["managed_vector_network_call_completed"] is False
    assert payload["patient_data_processed"] is False
    assert payload["clinical_validation"] is False
    assert payload["healthcare_production_ready"] is False


def test_unbound_port_fails_static_readiness(tmp_path: Path):
    compose = tmp_path / "compose.yml"
    compose.write_text(
        """
services:
  postgres: {ports: ["5432:5432"]}
  redis: {ports: ["127.0.0.1:6379:6379"]}
  backend: {ports: ["127.0.0.1:8017:8017"]}
  worker: {}
  frontend: {ports: ["127.0.0.1:5173:5173"]}
  n8n: {ports: ["127.0.0.1:5678:5678"]}
  mailhog: {ports: ["127.0.0.1:8025:8025"]}
volumes: {}
""",
        encoding="utf-8",
    )
    payload = build_disposable_synthetic_staging_readiness(
        root=tmp_path, compose_path=compose
    )
    assert payload["status"] == "needs_attention"
    check = next(
        row
        for row in payload["checks"]
        if row["check_id"] == "all_published_ports_loopback_only"
    )
    assert check["passed"] is False


def test_runtime_observations_are_recorded_without_expanding_claims():
    observations = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "runtime_started": True,
        "runtime_healthchecks_completed": True,
        "postgres_restore_drill_completed": True,
        "n8n_workflow_import_completed": True,
        "mailhog_delivery_receipt_completed": True,
        "real_external_delivery_completed": False,
        "patient_data_processed": False,
    }
    payload = build_disposable_synthetic_staging_readiness(
        runtime_observations=observations
    )
    assert payload["runtime_started"] is True
    assert payload["runtime_healthchecks_completed"] is True
    assert payload["postgres_restore_drill_completed"] is True
    assert payload["n8n_workflow_import_completed"] is True
    assert payload["mailhog_delivery_receipt_completed"] is True
    assert payload["real_external_delivery_completed"] is False
    assert payload["patient_data_processed"] is False
    assert payload["clinical_validation"] is False
    assert payload["healthcare_production_ready"] is False


def test_static_refresh_preserves_fresh_runtime_for_matching_compose(tmp_path: Path):
    output = tmp_path / "readiness.json"
    observations = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "runtime_started": True,
        "runtime_healthchecks_completed": True,
        "postgres_restore_drill_completed": True,
        "n8n_workflow_import_completed": True,
        "mailhog_delivery_receipt_completed": True,
    }
    first = write_disposable_synthetic_staging_readiness(
        output_path=output,
        runtime_observations=observations,
    )
    second = write_disposable_synthetic_staging_readiness(output_path=output)

    assert first["compose_sha256"] == second["compose_sha256"]
    assert second["runtime_started"] is True
    assert second["runtime_evidence_source"] == "preserved_matching_compose"
    assert second["postgres_restore_drill_completed"] is True
