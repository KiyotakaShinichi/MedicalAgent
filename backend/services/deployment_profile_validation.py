"""Validate runtime configuration without exposing secret values."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse


OUTPUT_PATH = Path("Data/evals/ops/latest_deployment_profile_validation.json")
UNSAFE_PASSWORDS = {"", "password", "postgres", "medical_agent", "change_me", "change_me_for_nonlocal_demo"}
PLACEHOLDER_SECRETS = {"", "replace_with_a_long_random_secret", "change_me", "secret"}


def _bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _check(name: str, ok: bool, reason: str) -> dict[str, object]:
    return {"name": name, "ok": bool(ok), "reason": reason}


def build_report(environment: Mapping[str, str] | None = None) -> dict[str, object]:
    import os
    from backend.services.oidc_auth import OIDCAuthError, load_oidc_config, validate_oidc_config

    env = dict(os.environ if environment is None else environment)
    profile = (env.get("ENVIRONMENT") or env.get("APP_ENV") or "development").strip().lower()
    strict = profile in {"staging", "production"}
    database_url = env.get("DATABASE_URL", "")
    redis_url = env.get("REDIS_URL", "")
    cors_value = env.get("NLCARE_CORS_ORIGINS") or env.get("ONCOTRACK_CORS_ORIGINS", "")
    cors = [part.strip() for part in cors_value.split(",") if part.strip()]
    dispatch_enabled = _bool(env.get("N8N_WEBHOOK_DISPATCH_ENABLED"))
    dispatch_url = env.get("N8N_WEBHOOK_BASE_URL", "")
    dispatch_secret = env.get("N8N_WEBHOOK_SIGNING_SECRET", "")
    parsed = urlparse(database_url.replace("postgresql+psycopg2", "postgresql")) if database_url else None
    password = parsed.password if parsed else None
    try:
        oidc_config = load_oidc_config(env)
        oidc_issues = validate_oidc_config(oidc_config, strict=strict)
        oidc_enabled = oidc_config.enabled
    except OIDCAuthError:
        oidc_enabled = _bool(env.get("NLCARE_OIDC_ENABLED"))
        oidc_issues = ["OIDC configuration could not be parsed"]

    checks = [
        _check(
            "database_backend",
            (not strict) or database_url.startswith(("postgresql://", "postgresql+psycopg2://")),
            "staging/production requires PostgreSQL; development may use SQLite",
        ),
        _check(
            "database_password_not_placeholder",
            (not strict) or (password is not None and password.lower() not in UNSAFE_PASSWORDS),
            "database password must be supplied and must not be a known placeholder",
        ),
        _check(
            "demo_auth_disabled",
            (not strict) or not _bool(env.get("ALLOW_DEMO_AUTH")),
            "demo credentials must be disabled in staging/production",
        ),
        _check(
            "cors_allowlist_explicit",
            (not strict) or (bool(cors) and "*" not in cors),
            "staging/production requires a non-wildcard CORS allowlist",
        ),
        _check(
            "production_cors_uses_https",
            profile != "production" or all(origin.startswith("https://") for origin in cors),
            "production browser origins must use HTTPS",
        ),
        _check(
            "production_cors_excludes_localhost",
            profile != "production" or all("localhost" not in origin and "127.0.0.1" not in origin for origin in cors),
            "production browser origins must not resolve to local development hosts",
        ),
        _check(
            "redis_configured",
            (not strict) or redis_url.startswith(("redis://", "rediss://")),
            "staging/production requires Redis for background engineering jobs",
        ),
        _check(
            "latest_migration_present",
            Path("backend/migrations/versions/0010_automation_task_leasing.py").exists(),
            "the latest automation leasing migration must ship with the image",
        ),
        _check(
            "non_demo_identity_provider_integrated",
            (not strict) or (oidc_enabled and not oidc_issues),
            "strict profiles require the feature-flagged OIDC issuer, audience, HTTPS JWKS URL, and RS256 validation",
        ),
        _check(
            "external_dispatch_uses_https",
            (not dispatch_enabled) or dispatch_url.startswith("https://"),
            "enabled n8n dispatch must use an HTTPS webhook endpoint",
        ),
        _check(
            "external_dispatch_secret_strong",
            (not dispatch_enabled)
            or (len(dispatch_secret) >= 32 and dispatch_secret.strip().lower() not in PLACEHOLDER_SECRETS),
            "enabled n8n dispatch requires a non-placeholder signing secret of at least 32 characters",
        ),
        _check(
            "external_dispatch_test_recipient_only",
            (not dispatch_enabled) or _bool(env.get("NLCARE_ALERT_TEST_RECIPIENT_ONLY")),
            "prototype external dispatch must remain restricted to synthetic test recipients",
        ),
    ]
    failed = [check for check in checks if not check["ok"]]
    status = "development_profile" if not strict else ("strong" if not failed else "blocked")
    return {
        "schema_version": "deployment_profile_validation_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "profile": profile,
        "strict_profile": strict,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "secrets_included_in_artifact": False,
        "non_demo_auth_implemented": True,
        "oidc_enabled": oidc_enabled,
        "oidc_config_valid": bool(oidc_enabled and not oidc_issues),
        "oidc_config_issue_count": len(oidc_issues),
        "external_dispatch_enabled": dispatch_enabled,
        "deployment_capability": (
            "local_or_controlled_demo_only"
            if not strict
            else ("strict_profile_configured_not_healthcare_ready" if not failed else "strict_profile_blocked")
        ),
        "n_checks": len(checks),
        "n_failed": len(failed),
        "checks": checks,
        "failed_checks": failed,
        "claim_boundary": (
            "Runtime configuration validation for a deployment-shaped engineering prototype. "
            "It is not a security certification, HIPAA claim, clinical validation, or evidence "
            "of production healthcare readiness."
        ),
    }


def write_report(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_report(), indent=2), encoding="utf-8")
    return output_path


__all__ = ["OUTPUT_PATH", "build_report", "write_report"]
