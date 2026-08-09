"""Validate runtime configuration without exposing secret values."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse


OUTPUT_PATH = Path("Data/evals/ops/latest_deployment_profile_validation.json")
MATRIX_OUTPUT_PATH = Path("Data/evals/ops/latest_deployment_profile_matrix.json")
UNSAFE_PASSWORDS = {"", "password", "postgres", "medical_agent", "change_me", "change_me_for_nonlocal_demo"}
PLACEHOLDER_SECRETS = {"", "replace_with_a_long_random_secret", "change_me", "secret"}


def _bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _check(name: str, ok: bool, reason: str) -> dict[str, object]:
    return {"name": name, "ok": bool(ok), "reason": reason}


def build_report(environment: Mapping[str, str] | None = None) -> dict[str, object]:
    import os
    from backend.services.oidc_auth import OIDCAuthError, load_oidc_config, validate_oidc_config
    from backend.services.oidc_pkce import load_browser_oidc_config, validate_browser_oidc_config
    from backend.services.upload_security import UploadSecurityPolicy, validate_upload_security_policy

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
    synthetic_only = _bool(env.get("NLCARE_SYNTHETIC_ONLY"))
    data_classification = str(env.get("NLCARE_DATA_CLASSIFICATION") or "").strip().lower()
    uploads_enabled = _bool(env.get("NLCARE_UPLOADS_ENABLED"))
    upload_scanner_mode = str(env.get("NLCARE_UPLOAD_SCANNER_MODE") or "disabled").strip().lower()
    upload_scanner_command = str(env.get("NLCARE_UPLOAD_SCANNER_COMMAND") or "").strip()
    parsed = urlparse(database_url.replace("postgresql+psycopg2", "postgresql")) if database_url else None
    password = parsed.password if parsed else None
    try:
        oidc_config = load_oidc_config(env)
        oidc_issues = validate_oidc_config(oidc_config, strict=strict)
        oidc_enabled = oidc_config.enabled
    except OIDCAuthError:
        oidc_enabled = _bool(env.get("NLCARE_OIDC_ENABLED"))
        oidc_issues = ["OIDC configuration could not be parsed"]
    browser_oidc_issues = validate_browser_oidc_config(load_browser_oidc_config(env), strict=strict)
    try:
        validate_upload_security_policy(
            UploadSecurityPolicy(
                enabled=uploads_enabled,
                strict_profile=strict,
                scanner_mode=upload_scanner_mode,
                scanner_command=upload_scanner_command,
                scanner_timeout_seconds=20,
            )
        )
        upload_policy_valid = True
    except ValueError:
        upload_policy_valid = False
    frontend_sources = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in Path("frontend-react/src").rglob("*.ts*")
    )
    local_storage_bearer_markers = [
        marker for marker in ("localStorage.getItem(\"patientPortalAccessToken\")", "localStorage.setItem(TOKEN_KEYS")
        if marker in frontend_sources
    ]

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
            "browser_oidc_pkce_configured",
            (not strict) or not browser_oidc_issues,
            "strict profiles require HTTPS browser authorization/token endpoints, client ID, redirect URI, and openid scope",
        ),
        _check(
            "bearer_tokens_not_persisted_in_local_storage",
            not local_storage_bearer_markers,
            "browser bearer tokens must not be persisted in localStorage",
        ),
        _check(
            "synthetic_only_runtime_lock",
            (not strict) or synthetic_only,
            "restricted staging/production-shaped profiles must enforce synthetic-only operation",
        ),
        _check(
            "synthetic_data_classification",
            (not strict) or data_classification == "synthetic",
            "strict profiles must label the accepted data class as synthetic",
        ),
        _check(
            "upload_quarantine_policy",
            (not strict) or (not uploads_enabled) or upload_policy_valid,
            "strict-profile uploads must be disabled or use a configured external fail-closed scanner",
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
        "schema_version": "deployment_profile_validation_v3",
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
        "browser_oidc_config_valid": not browser_oidc_issues,
        "browser_oidc_issue_count": len(browser_oidc_issues),
        "synthetic_only": synthetic_only,
        "data_classification": data_classification or "unset",
        "uploads_enabled": uploads_enabled,
        "upload_policy_valid": upload_policy_valid,
        "local_storage_bearer_marker_count": len(local_storage_bearer_markers),
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


def build_profile_matrix(
    active_environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    active = build_report(active_environment)
    strict_valid = build_report(_production_shaped_environment())
    strict_invalid = build_report(_unsafe_production_environment())
    fail_closed = bool(
        strict_invalid["status"] == "blocked"
        and int(strict_invalid["n_failed"]) >= 4
    )
    matrix_pass = bool(
        active["status"] != "blocked"
        and strict_valid["status"] == "strong"
        and fail_closed
        and strict_valid["healthcare_production_ready"] is False
    )
    return {
        "schema_version": "deployment_profile_matrix_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if matrix_pass else "needs_attention",
        "active_profile": {
            "profile": active["profile"],
            "status": active["status"],
            "strict_profile": active["strict_profile"],
            "deployment_capability": active["deployment_capability"],
            "n_failed": active["n_failed"],
        },
        "production_shaped_static_validation": {
            "status": strict_valid["status"],
            "n_checks": strict_valid["n_checks"],
            "n_failed": strict_valid["n_failed"],
            "oidc_config_valid": strict_valid["oidc_config_valid"],
            "uses_non_resolving_example_endpoints": True,
            "live_identity_provider_contacted": False,
            "live_database_contacted": False,
            "live_redis_contacted": False,
        },
        "unsafe_profile_fail_closed": {
            "status": strict_invalid["status"],
            "n_failed": strict_invalid["n_failed"],
            "failed_check_names": [
                str(item["name"]) for item in strict_invalid["failed_checks"]
            ],
            "passed": fail_closed,
        },
        "matrix_passed": matrix_pass,
        "cloud_deployment_completed": False,
        "live_oidc_integration_completed": False,
        "production_traffic_observed": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "secrets_included_in_artifact": False,
        "deployment_capability": "configuration_matrix_validated_local_runtime_only",
        "claim_boundary": (
            "This matrix proves static configuration checks and fail-closed behavior for "
            "an engineering prototype. It does not prove cloud deployment, live identity "
            "integration, security certification, compliance, clinical validation, or "
            "production healthcare readiness."
        ),
    }


def write_profile_matrix(
    output_path: Path = MATRIX_OUTPUT_PATH,
    *,
    active_environment: Mapping[str, str] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_profile_matrix(active_environment), indent=2),
        encoding="utf-8",
    )
    return output_path


def _production_shaped_environment() -> dict[str, str]:
    return {
        "ENVIRONMENT": "production",
        "DATABASE_URL": (
            "postgresql+psycopg2://nlcare:"
            "static-validation-password-42@db.nlcare.invalid/nlcare"
        ),
        "REDIS_URL": "rediss://cache.nlcare.invalid:6380/0",
        "ALLOW_DEMO_AUTH": "false",
        "NLCARE_CORS_ORIGINS": "https://app.nlcare.invalid",
        "NLCARE_OIDC_ENABLED": "true",
        "NLCARE_OIDC_ISSUER": "https://identity.nlcare.invalid",
        "NLCARE_OIDC_AUDIENCE": "nlcare-api",
        "NLCARE_OIDC_JWKS_URL": "https://identity.nlcare.invalid/.well-known/jwks.json",
        "NLCARE_OIDC_ALGORITHMS": "RS256",
        "NLCARE_OIDC_AUTHORIZATION_ENDPOINT": "https://identity.nlcare.invalid/authorize",
        "NLCARE_OIDC_TOKEN_ENDPOINT": "https://identity.nlcare.invalid/token",
        "NLCARE_OIDC_CLIENT_ID": "nlcare-browser",
        "NLCARE_OIDC_REDIRECT_URI": "https://app.nlcare.invalid/auth/callback",
        "NLCARE_OIDC_SCOPES": "openid profile",
        "NLCARE_SYNTHETIC_ONLY": "true",
        "NLCARE_DATA_CLASSIFICATION": "synthetic",
        "NLCARE_UPLOADS_ENABLED": "false",
        "NLCARE_UPLOAD_SCANNER_MODE": "disabled",
        "N8N_WEBHOOK_DISPATCH_ENABLED": "false",
    }


def _unsafe_production_environment() -> dict[str, str]:
    return {
        "ENVIRONMENT": "production",
        "DATABASE_URL": "sqlite:///unsafe.db",
        "ALLOW_DEMO_AUTH": "true",
        "NLCARE_CORS_ORIGINS": "*",
        "NLCARE_OIDC_ENABLED": "false",
        "NLCARE_SYNTHETIC_ONLY": "false",
        "NLCARE_DATA_CLASSIFICATION": "unknown",
        "NLCARE_UPLOADS_ENABLED": "true",
        "NLCARE_UPLOAD_SCANNER_MODE": "builtin",
        "N8N_WEBHOOK_DISPATCH_ENABLED": "true",
        "N8N_WEBHOOK_BASE_URL": "http://127.0.0.1:5678/webhook/nlcare",
        "N8N_WEBHOOK_SIGNING_SECRET": "change_me",
        "NLCARE_ALERT_TEST_RECIPIENT_ONLY": "false",
    }


__all__ = [
    "MATRIX_OUTPUT_PATH",
    "OUTPUT_PATH",
    "build_profile_matrix",
    "build_report",
    "write_profile_matrix",
    "write_report",
]
