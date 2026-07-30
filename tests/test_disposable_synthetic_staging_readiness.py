from __future__ import annotations

from pathlib import Path

from backend.services.disposable_synthetic_staging_readiness import (
    REQUIRED_SERVICES,
    build_disposable_synthetic_staging_readiness,
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
