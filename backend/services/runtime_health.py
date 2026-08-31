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


# A liveness probe that can block is itself a restart vector: an orchestrator
# that times out waiting for /health restarts the process just as surely as a
# 500 would. The database probe below is therefore bounded, and the bound is
# deliberately far shorter than any sane probe timeout.
LIVENESS_DATABASE_PROBE_TIMEOUT_SECONDS = 0.5


def database_connectivity(
    db: Session, *, timeout_seconds: float = LIVENESS_DATABASE_PROBE_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """Bounded, informational answer to "can we reach the database right now?"

    Never raises and never blocks for longer than `timeout_seconds`. The query
    runs on a worker thread so that a driver call which hangs - rather than
    failing - cannot hold the liveness response open; if the deadline passes the
    answer is simply "not connected", and the abandoned thread dies with its
    session.

    Reports `error_type` only, never the exception message. Connection failures
    routinely carry the DSN, and this endpoint is unauthenticated.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

    def probe() -> None:
        db.execute(text("SELECT 1"))

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        executor.submit(probe).result(timeout=timeout_seconds)
        return {"connected": True}
    except FutureTimeout:
        return {"connected": False, "error_type": "TimeoutError"}
    except Exception as exc:  # noqa: BLE001 - liveness must survive any failure
        return {"connected": False, "error_type": type(exc).__name__}
    finally:
        # Do not wait: the point of the timeout is to not block on a hung probe.
        executor.shutdown(wait=False)


def rag_index_liveness() -> dict[str, Any]:
    """Whether the retrieval index is loaded in this process. Informational.

    Reads the in-process cache counters only. It never loads, builds, or warms
    an index: doing that from a liveness probe would turn a cheap health check
    into the most expensive request the service handles, and an orchestrator
    polling it every few seconds would keep paying that cost.

    `loaded: false` is therefore a normal answer, not a fault. It means this
    process has not served a retrieval query yet - a freshly started replica
    reports false until its first search or its prewarm finishes. Whether
    retrieval is *ready enough to serve traffic* is a different question, and
    `/ready` answers it authoritatively.

    Never raises: a liveness probe must not fail because an optional subsystem
    could not be inspected.
    """
    try:
        from backend.services.rag_vector_index import rag_runtime_cache_stats

        stats = rag_runtime_cache_stats()
        return {"loaded": int(stats.get("cached_index_count", 0)) > 0}
    except Exception as exc:  # noqa: BLE001 - liveness must survive any failure
        return {"loaded": False, "error_type": type(exc).__name__}


def liveness_payload(
    database: Mapping[str, Any] | None = None,
    rag_index: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Liveness answer, with database reachability reported for visibility.

    Reports the running version alongside status so an operator can tell *which
    build* answered, which is the difference between a useful health probe and
    one that only proves a socket is open.

    `database` is **informational and non-authoritative**. It never changes
    `status`, and the route never changes its HTTP code because of it: a live
    process with an unreachable database is still live, and restarting it would
    not fix the database. `/ready` remains the authoritative, fail-closed
    signal that decides whether traffic should be routed here.

    `rag_index` is informational on the same terms: it reports whether the
    retrieval index is loaded in this process, and `loaded: false` is a normal
    state for a replica that has not served a query yet.

    Callers pass the results of `database_connectivity` and
    `rag_index_liveness`; when no probe was run the key is reported as unprobed
    rather than guessed.
    """
    return {
        "status": "ok",
        "service": "nlcare_monitoring_prototype",
        "version": application_version(),
        "database": dict(database)
        if database is not None
        else {"connected": False, "error_type": "NotProbed"},
        "rag_index": dict(rag_index)
        if rag_index is not None
        else {"loaded": False, "error_type": "NotProbed"},
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
        checks["database"] = {"ready": True, "required": True}
    except Exception as exc:
        checks["database"] = {
            "ready": False,
            "required": True,
            "error_type": type(exc).__name__,
        }

    try:
        retrieval = retrieval_probe() if retrieval_probe else {}
        retrieval_ready = bool(retrieval.get("meets_deployment_requirement"))
        checks["retrieval"] = {
            "ready": retrieval_ready,
            "required": True,
            "summary": _bounded_retrieval_summary(retrieval),
        }
    except Exception as exc:
        checks["retrieval"] = {
            "ready": False,
            "required": True,
            "error_type": type(exc).__name__,
        }

    shared_redis_required = str(env.get("NLCARE_SHARED_RATE_LIMIT_ENABLED", "")).lower() in {
        "1", "true", "yes", "on",
    }
    if shared_redis_required:
        checks["redis"] = _redis_readiness(str(env.get("REDIS_URL", "")))
    else:
        checks["redis"] = {"ready": True, "required": False}

    try:
        demo_auth_allowed = bool(demo_auth_probe()) if demo_auth_probe else False
    except Exception as exc:
        demo_auth_allowed = False
        checks["configuration"] = {
            "ready": False,
            "required": True,
            "error_type": type(exc).__name__,
        }

    ready = all(
        bool(check.get("ready"))
        for check in checks.values()
        if bool(check.get("required", True))
    )
    app_environment = str(env.get("ENVIRONMENT") or env.get("APP_ENV") or "development")
    return {
        "status": "ready" if ready else "not_ready",
        "service": "nlcare_monitoring_prototype",
        "environment": app_environment.strip().lower(),
        "demo_auth_allowed": demo_auth_allowed,
        "checks": checks,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "claim_boundary": (
            "Readiness means engineering dependencies answered bounded probes. "
            "It is not clinical validation, real-patient approval, or PHI compliance."
        ),
    }, ready


def _bounded_retrieval_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    """Expose readiness facts without paths, URLs, content, or credentials."""
    allowed = (
        "status",
        "backend",
        "dense_backend_required",
        "dense_backend_active",
        "meets_deployment_requirement",
        "active_mode",
    )
    return {key: value[key] for key in allowed if key in value}


def _redis_readiness(url: str) -> dict[str, Any]:
    if not url:
        return {"ready": False, "required": True, "error_type": "MissingRedisUrl"}
    try:
        from redis import Redis

        client = Redis.from_url(url, socket_connect_timeout=0.25, socket_timeout=0.25)
        return {"ready": bool(client.ping()), "required": True}
    except Exception as exc:
        return {"ready": False, "required": True, "error_type": type(exc).__name__}


__all__ = [
    "application_version",
    "database_connectivity",
    "rag_index_liveness",
    "liveness_payload",
    "readiness_payload",
]
