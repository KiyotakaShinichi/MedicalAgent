from __future__ import annotations

from pathlib import Path

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
    assert "127.0.0.1:55432" in content
    assert "127.0.0.1:56379" in content
    assert "medical-agent-postgres" not in content
