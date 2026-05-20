from __future__ import annotations

import pytest

from backend.services.auth import _ensure_demo_auth_allowed


def test_demo_auth_is_disabled_in_production_without_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("ALLOW_DEMO_AUTH", raising=False)

    with pytest.raises(ValueError, match="Demo authentication is disabled"):
        _ensure_demo_auth_allowed()


def test_demo_auth_allowed_in_development_by_default(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.delenv("ALLOW_DEMO_AUTH", raising=False)

    _ensure_demo_auth_allowed()
