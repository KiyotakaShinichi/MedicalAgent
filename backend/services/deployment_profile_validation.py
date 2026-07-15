"""Validate runtime configuration without exposing secret values."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse


OUTPUT_PATH = Path("Data/evals/ops/latest_deployment_profile_validation.json")
UNSAFE_PASSWORDS = {"", "password", "postgres", "medical_agent", "change_me", "change_me_for_nonlocal_demo"}


def _bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _check(name: str, ok: bool, reason: str) -> dict[str, object]:
    return {"name": name, "ok": bool(ok), "reason": reason}


def build_report(environment: Mapping[str, str] | None = None) -> dict[str, object]:
    import os

    env = dict(os.environ if environment is None else environment)
    profile = (env.get("ENVIRONMENT") or env.get("APP_ENV") or "development").strip().lower()
    strict = profile in {"staging", "production"}
    database_url = env.get("DATABASE_URL", "")
    redis_url = env.get("REDIS_URL", "")
    cors = [part.strip() for part in env.get("ONCOTRACK_CORS_ORIGINS", "").split(",") if part.strip()]
    parsed = urlparse(database_url.replace("postgresql+psycopg2", "postgresql")) if database_url else None
    password = parsed.password if parsed else None

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
            "redis_configured",
            (not strict) or redis_url.startswith(("redis://", "rediss://")),
            "staging/production requires Redis for background engineering jobs",
        ),
        _check(
            "migration_present",
            Path("backend/migrations/versions/0007_confirmed_record_writes.py").exists(),
            "confirmed-write audit migration must ship with the image",
        ),
    ]
    failed = [check for check in checks if not check["ok"]]
    status = "development_profile" if not strict else ("strong" if not failed else "blocked")
    return {
        "schema_version": "deployment_profile_validation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "profile": profile,
        "strict_profile": strict,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "secrets_included_in_artifact": False,
        "n_checks": len(checks),
        "n_failed": len(failed),
        "checks": checks,
        "failed_checks": failed,
        "claim_boundary": (
            "Runtime configuration validation for a production-shaped engineering prototype. "
            "It is not a security certification, HIPAA claim, clinical validation, or evidence "
            "of production healthcare readiness."
        ),
    }


def write_report(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_report(), indent=2), encoding="utf-8")
    return output_path


__all__ = ["OUTPUT_PATH", "build_report", "write_report"]
