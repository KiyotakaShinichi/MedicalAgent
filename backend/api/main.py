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

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.schema_migrations import ensure_schema

from backend.api.deps import get_admin_access_context, get_db
from backend.api.routers import health
from backend.api.routers.auth import router as auth_router
from backend.api.routers.patient import router as patient_router, warm_patient_report_enrichment_cache
from backend.api.routers.clinician_review import router as clinician_review_router
from backend.api.routers.admin import router as admin_router
from backend.api.routers.model import router as model_router
from backend.api.routers.admin_eval import build_admin_eval_router
from backend.api.routers.automation import router as automation_router
from backend.api.routers.platform import router as platform_router
from backend.services.request_context import get_request_id, reset_request_id, set_request_id
from backend.services.api_protection import EngineeringApiProtectionMiddleware
from backend.services.synthetic_data_boundary import SyntheticDataBoundaryMiddleware
from backend.services.llm_telemetry import reset_llm_telemetry, start_llm_telemetry
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

# Operational probes. Registered as a router so `/health` and `/ready` are
# discoverable the same way every other route group is, rather than being
# decorated inline on the app object.
app.include_router(health.router)
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




# ─── Static files ─────────────────────────────────────────────────────────────

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
