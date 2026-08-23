"""Cheap liveness and bounded dependency-readiness probes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping

from sqlalchemy import text
from sqlalchemy.orm import Session


def application_version() -> str:
    """Deployed application version, or `unknown` when it cannot be resolved.

    Read from installed package metadata so the value reflects what is actually
    running rather than a constant that can drift from the build. Never raises:
    a liveness probe must not fail because version lookup did.
    """
    try:
        from importlib.metadata import version

        return version("nlcare-medical-agent")
    except Exception:  # noqa: BLE001 - liveness must not depend on metadata
        pass
    try:
        import tomllib

        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        return str(tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"])
    except Exception:  # noqa: BLE001
        return "unknown"


def liveness_payload() -> dict[str, Any]:
    """Cheap liveness answer.

    Reports the running version alongside status so an operator can tell *which
    build* answered, which is the difference between a useful health probe and
    one that only proves a socket is open. Still touches no database, cache, or
    network dependency - dependency state belongs to `readiness_payload`.
    """
    return {
        "status": "ok",
        "service": "nlcare_monitoring_prototype",
        "version": application_version(),
    }


def readiness_payload(
    db: Session,
    *,
    environment: Mapping[str, str] | None = None,
    retrieval_probe: Callable[[], dict[str, Any]] | None = None,
    demo_auth_probe: Callable[[], bool] | None = None,
) -> tuple[dict[str, Any], bool]:
    env = os.environ if environment is None else environment
    checks: dict[str, Any] = {}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = {"ready": True}
    except Exception as exc:
        checks["database"] = {"ready": False, "error_type": type(exc).__name__}

    try:
        retrieval = retrieval_probe() if retrieval_probe else {}
        retrieval_ready = bool(retrieval.get("meets_deployment_requirement"))
        checks["retrieval"] = {"ready": retrieval_ready, "summary": retrieval}
    except Exception as exc:
        checks["retrieval"] = {"ready": False, "error_type": type(exc).__name__}

    shared_redis_required = str(env.get("NLCARE_SHARED_RATE_LIMIT_ENABLED", "")).lower() in {
        "1", "true", "yes", "on",
    }
    if shared_redis_required:
        checks["redis"] = _redis_readiness(str(env.get("REDIS_URL", "")))
    else:
        checks["redis"] = {"ready": True, "required": False}

    ready = all(bool(check.get("ready")) for check in checks.values())
    app_environment = str(env.get("ENVIRONMENT") or env.get("APP_ENV") or "development")
    return {
        "status": "ready" if ready else "not_ready",
        "service": "nlcare_monitoring_prototype",
        "environment": app_environment.strip().lower(),
        "demo_auth_allowed": bool(demo_auth_probe()) if demo_auth_probe else False,
        "checks": checks,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "claim_boundary": (
            "Readiness means engineering dependencies answered bounded probes. "
            "It is not clinical validation, real-patient approval, or PHI compliance."
        ),
    }, ready


def _redis_readiness(url: str) -> dict[str, Any]:
    if not url:
        return {"ready": False, "required": True, "error_type": "MissingRedisUrl"}
    try:
        from redis import Redis

        client = Redis.from_url(url, socket_connect_timeout=0.25, socket_timeout=0.25)
        return {"ready": bool(client.ping()), "required": True}
    except Exception as exc:
        return {"ready": False, "required": True, "error_type": type(exc).__name__}


__all__ = ["application_version", "liveness_payload", "readiness_payload"]
