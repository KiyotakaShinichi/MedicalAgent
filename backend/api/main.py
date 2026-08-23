"""
AI Breast Cancer Monitoring System — FastAPI application entrypoint.

This file wires together routers, middleware, static file mounts, and the
health-check / redirect routes. All business logic lives in routers/.
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from backend.schema_migrations import ensure_schema

from backend.api.deps import get_admin_access_context, get_db
from backend.api.routers.auth import router as auth_router
from backend.api.routers.patient import router as patient_router, warm_patient_report_enrichment_cache
from backend.api.routers.clinician_review import router as clinician_review_router
from backend.api.routers.admin import router as admin_router
from backend.api.routers.model import router as model_router
from backend.api.routers.admin_eval import build_admin_eval_router
from backend.api.routers.automation import router as automation_router
from backend.api.routers.platform import router as platform_router
from backend.api.schemas.operations import LivenessResponse, ReadinessResponse
from backend.services.request_context import get_request_id, reset_request_id, set_request_id
from backend.services.api_protection import EngineeringApiProtectionMiddleware
from backend.services.synthetic_data_boundary import SyntheticDataBoundaryMiddleware
from backend.services.llm_telemetry import reset_llm_telemetry, start_llm_telemetry
from backend.services.runtime_health import (
    database_connectivity,
    liveness_payload,
    readiness_payload,
)
from backend.logging_config import configure_logging, log_event


# ─── App setup ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_: FastAPI):
    warm_patient_report_enrichment_cache()
    prewarm_task = None
    environment = (os.environ.get("ENVIRONMENT") or os.environ.get("APP_ENV") or "development").strip().lower()
    prewarm_enabled = os.environ.get("NLCARE_RAG_PREWARM", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }
    if prewarm_enabled and environment != "test" and not os.environ.get("PYTEST_CURRENT_TEST"):
        from backend.services.agent_kb_corpus import get_rag_corpus, knowledge_base_fingerprint
        from backend.services.rag_vector_index import prewarm_rag_vector_runtime

        corpus = get_rag_corpus()
        fingerprint = knowledge_base_fingerprint()
        prewarm_task = asyncio.create_task(
            asyncio.to_thread(
                prewarm_rag_vector_runtime,
                corpus,
                knowledge_fingerprint=fingerprint,
            )
        )
    yield
    if prewarm_task is not None and not prewarm_task.done():
        prewarm_task.cancel()


# Declared so the operational probes are discoverable as a group rather than
# appearing as two untagged operations among ~200 clinical ones.
OPENAPI_TAGS = [
    {
        "name": "operations",
        "description": (
            "Liveness and readiness probes for deployment orchestration. "
            "No patient data, no clinical meaning, no authentication required."
        ),
    },
]

# Structured JSON logging is installed before anything else can emit a
# record. See backend/logging_config.py for the pipeline and its redaction
# policy; configure_logging() is idempotent, so this is safe on re-import.
app = FastAPI(
    title="NLCare Breast Cancer Monitoring Engineering Prototype",
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
)
configure_logging()
ensure_schema()

# CORS — explicit origin list.  FastAPI/Starlette warns that the combination
# ``allow_origins=["*"] + allow_credentials=True`` is unsafe; browsers also
# refuse credentialed requests against a wildcard.  Override via the
# ``ONCOTRACK_CORS_ORIGINS`` env var (comma-separated) for non-default
# deployments — e.g. a staging frontend at https://app.example.com.
_DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8017",
    "http://127.0.0.1:8017",
]
_cors_env = os.environ.get("NLCARE_CORS_ORIGINS") or os.environ.get("ONCOTRACK_CORS_ORIGINS")
_cors_origins = (
    [origin.strip() for origin in _cors_env.split(",") if origin.strip()]
    if _cors_env
    else _DEFAULT_CORS_ORIGINS
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-Request-ID",
        "X-NLCare-Receipt-Signature",
        "X-NLCare-Timestamp",
        "X-NLCare-Data-Class",
        "X-NLCare-Organization-ID",
        "Idempotency-Key",
    ],
    expose_headers=["X-Request-ID", "X-Analytics-Cache"],
)
app.add_middleware(EngineeringApiProtectionMiddleware)
app.add_middleware(SyntheticDataBoundaryMiddleware)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid4())
    started = time.perf_counter()
    token = set_request_id(request_id)
    telemetry_token = start_llm_telemetry()
    try:
        response = await call_next(request)
    except Exception as exc:
        route = getattr(request.scope.get("route"), "path", "unmatched")
        log_event(
            "http_request_failed",
            severity="error",
            request_id=request_id,
            component="api",
            details={
                "method": request.method,
                "route": route,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "error_type": type(exc).__name__,
            },
        )
        raise
    finally:
        reset_llm_telemetry(telemetry_token)
        reset_request_id(token)
    route = getattr(request.scope.get("route"), "path", "unmatched")
    log_event(
        "http_request_completed",
        severity="warning" if response.status_code >= 400 else "info",
        request_id=request_id,
        component="api",
        details={
            "method": request.method,
            "route": route,
            "status_code": response.status_code,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        },
    )
    response.headers["x-request-id"] = request_id
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'self'",
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a correlated, non-revealing 500 instead of a bare stack trace.

    Without this, an unhandled exception produced a plain Starlette 500 whose
    body carried no correlation id, so a user-reported failure could not be
    tied to a log line. The middleware's `X-Request-ID` header never made it
    onto that response either, because the exception unwound past the code
    that sets it.

    What is deliberately *not* here: the exception message, its traceback, the
    failing module, or any request content. Those go to the structured log
    under the same `request_id`; the response carries the id and nothing more,
    so a caller can quote it in a bug report without the API having disclosed
    connection strings, file paths, or prompt text.

    `HTTPException` is unaffected — FastAPI's own handler still owns it, so
    404s and 403s keep their existing bodies.
    """
    request_id = get_request_id() or request.headers.get("x-request-id") or str(uuid4())
    route = getattr(request.scope.get("route"), "path", "unmatched")
    log_event(
        "http_request_unhandled_exception",
        severity="error",
        request_id=request_id,
        component="api",
        details={
            "method": request.method,
            "route": route,
            "error_type": type(exc).__name__,
        },
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "request_id": request_id,
            "detail": "The request failed. Quote the request_id when reporting this.",
        },
        headers={"X-Request-ID": request_id},
    )


# ─── Routers ──────────────────────────────────────────────────────────────────

app.include_router(auth_router)
app.include_router(patient_router)
app.include_router(clinician_review_router)
app.include_router(admin_router)
app.include_router(model_router)
app.include_router(build_admin_eval_router(get_admin_access_context, get_db))
app.include_router(automation_router)
app.include_router(platform_router)


# ─── Core routes ──────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/frontend/login.html")


@app.get("/login", include_in_schema=False)
def login_page():
    return RedirectResponse(url="/frontend/login.html")


@app.get("/patient", include_in_schema=False)
def patient_portal():
    return RedirectResponse(url="/frontend/patient.html")


@app.get("/clinician", include_in_schema=False)
def clinician_dashboard():
    return RedirectResponse(url="/frontend/index.html")


@app.get("/admin", include_in_schema=False)
def admin_dashboard():
    return RedirectResponse(url="/frontend/admin.html")


# The path stays on the decorator line, not wrapped onto its own, so that
# `@app.get("/health"` is greppable as a single string. Route registration is
# something people and tools look for by literal text.
@app.get("/health", tags=["operations"], summary="Liveness probe",
         response_model=LivenessResponse, operation_id="getHealth",
         responses={200: {"description": "Process is alive and able to serve requests."}})
@app.get("/healthz", tags=["operations"], include_in_schema=False, response_model=LivenessResponse)
def healthcheck(db: Session = Depends(get_db)):
    """Liveness probe. Returns 200 whenever the process can serve requests.

    Reports `status`, `service`, `version`, and database reachability.

    The database result is **informational**. This endpoint returns 200 for a
    live process even when the database is unreachable, and `status` stays
    `ok`: an orchestrator uses liveness to decide whether to *restart* the
    process, and restarting it cannot repair a database. Letting a dependency
    failure fail this probe is what turns a degraded database into a
    cluster-wide restart loop. `/ready` remains the authoritative, fail-closed
    signal that decides whether traffic should be routed here, and it is the
    one that returns 503.

    The probe itself is bounded (see `LIVENESS_DATABASE_PROBE_TIMEOUT_SECONDS`)
    because a liveness probe that *hangs* is a restart vector too. It reports
    an exception class name at most - never a message, host, or connection
    string, since this route is unauthenticated.

    `/healthz` is an unlisted alias for orchestrators that probe the
    Kubernetes-conventional path.
    """
    return liveness_payload(database_connectivity(db))


@app.get(
    "/ready",
    tags=["operations"],
    summary="Readiness probe",
    response_model=ReadinessResponse,
    operation_id="getReady",
    responses={
        200: {"description": "Every required dependency answered its probe."},
        503: {
            "description": "At least one required dependency is not ready.",
            "model": ReadinessResponse,
        },
    },
)
@app.get("/readyz", tags=["operations"], include_in_schema=False, response_model=ReadinessResponse)
def readinesscheck(response: Response, db: Session = Depends(get_db)):
    """Runtime readiness probe for engineering deployments.

    Checks database reachability, retrieval-index availability, and — only
    when shared rate limiting is enabled — Redis. Each probe is bounded, and a
    failing probe reports its exception *class name* only, never the message,
    which can carry connection strings or filesystem paths.

    Returns 503 when any required dependency is not ready, so a load balancer
    can drain the instance without restarting it.

    This reports deployment posture. It does not imply healthcare production
    readiness or clinical validation.

    `/readyz` is an unlisted alias for the Kubernetes-conventional path.
    """
    from backend.services.auth import is_demo_auth_allowed
    from backend.services.rag_vector_index import rag_runtime_readiness

    payload, ready = readiness_payload(
        db,
        retrieval_probe=rag_runtime_readiness,
        demo_auth_probe=is_demo_auth_allowed,
    )
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return payload


# ─── Static files ─────────────────────────────────────────────────────────────

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
