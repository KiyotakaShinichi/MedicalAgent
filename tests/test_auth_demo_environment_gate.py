from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.services.auth import _ensure_demo_auth_allowed, is_demo_auth_allowed
from backend.api.main import app


def test_demo_auth_is_disabled_in_production_without_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("ALLOW_DEMO_AUTH", raising=False)

    with pytest.raises(ValueError, match="Demo authentication is disabled"):
        _ensure_demo_auth_allowed()


def test_demo_auth_allowed_in_development_by_default(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.delenv("ALLOW_DEMO_AUTH", raising=False)

    _ensure_demo_auth_allowed()
    assert is_demo_auth_allowed() is True


def test_effective_demo_auth_posture_matches_production_opt_in(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("ALLOW_DEMO_AUTH", "true")
    assert is_demo_auth_allowed() is True


def test_demo_patient_listing_is_hidden_when_demo_auth_is_disabled(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("ALLOW_DEMO_AUTH", raising=False)

    response = TestClient(app, raise_server_exceptions=False).get("/auth/demo-patients")

    assert response.status_code == 404
    assert response.json()["detail"] == "Demo patient listing is disabled"
