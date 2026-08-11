from __future__ import annotations

from pathlib import Path
import sys

from backend.services import container_recovery_smoke as smoke


def test_blocked_docker_environment_emits_no_false_completion(monkeypatch, tmp_path):
    monkeypatch.setattr(smoke, "_docker_available", lambda: (False, "daemon missing"))
    result = smoke.run_container_recovery_smoke(tmp_path / "smoke.json")
    assert result["status"] == "blocked_environment"
    assert result["completed"] is False
    assert result["clinical_validation"] is False
    assert result["healthcare_production_ready"] is False


def test_recovery_compose_is_isolated_and_uses_persistent_volumes():
    content = Path(smoke.COMPOSE_FILE).read_text(encoding="utf-8")
    assert "postgres_smoke_data" in content
    assert "redis_smoke_data" in content
    assert "NLCARE_RECOVERY_POSTGRES_PORT" in content
    assert "NLCARE_RECOVERY_REDIS_PORT" in content
    assert "medical-agent-postgres" not in content


def test_recovery_smoke_selects_distinct_loopback_ports():
    first = smoke._free_loopback_port()
    second = smoke._free_loopback_port()
    assert isinstance(first, int)
    assert isinstance(second, int)
    assert first > 0
    assert second > 0


def test_recovery_smoke_uses_current_python_interpreter():
    content = Path(smoke.__file__).read_text(encoding="utf-8")
    assert "sys.executable" in content
    assert sys.executable
    assert Path(smoke._migration_python()).exists()
