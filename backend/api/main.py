"""
AI Breast Cancer Monitoring System — FastAPI application entrypoint.

This file wires together routers, middleware, static file mounts, and the
health-check / redirect routes. All business logic lives in routers/.
"""

import os
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from backend.database import SessionLocal
from backend.schema_migrations import ensure_schema

from backend.api.deps import get_access_context, get_admin_access_context, get_db
from backend.api.routers.auth import router as auth_router
from backend.api.routers.patient import router as patient_router, warm_patient_report_enrichment_cache
from backend.api.routers.clinician_review import router as clinician_review_router
from backend.api.routers.admin import router as admin_router
from backend.api.routers.model import router as model_router
from backend.api.routers.admin_eval import build_admin_eval_router
from backend.api.routers.automation import router as automation_router
from backend.services.request_context import reset_request_id, set_request_id
from backend.services.api_protection import EngineeringApiProtectionMiddleware
from backend.services.llm_telemetry import reset_llm_telemetry, start_llm_telemetry


# ─── App setup ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_: FastAPI):
    warm_patient_report_enrichment_cache()
    yield


app = FastAPI(
    title="NLCare Breast Cancer Monitoring Engineering Prototype",
    lifespan=lifespan,
)
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
    ],
    expose_headers=["X-Request-ID", "X-Analytics-Cache"],
)
app.add_middleware(EngineeringApiProtectionMiddleware)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid4())
    token = set_request_id(request_id)
    telemetry_token = start_llm_telemetry()
    try:
        response = await call_next(request)
    finally:
        reset_llm_telemetry(telemetry_token)
        reset_request_id(token)
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


# ─── Routers ──────────────────────────────────────────────────────────────────

app.include_router(auth_router)
app.include_router(patient_router)
app.include_router(clinician_review_router)
app.include_router(admin_router)
app.include_router(model_router)
app.include_router(build_admin_eval_router(get_admin_access_context, get_db))
app.include_router(automation_router)


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


@app.get("/health")
def healthcheck(db: Session = Depends(get_db)):
    from sqlalchemy import text

    db.execute(text("SELECT 1"))
    return {
        "status": "ok",
        "service": "nlcare_monitoring_prototype",
        "database": "ok",
    }


@app.get("/ready")
def readinesscheck(db: Session = Depends(get_db)):
    """Runtime readiness probe for engineering deployments.

    This checks database reachability and reports deployment posture. It does
    not imply healthcare production readiness or clinical validation.
    """
    from sqlalchemy import text

    db.execute(text("SELECT 1"))
    environment = (os.environ.get("ENVIRONMENT") or os.environ.get("APP_ENV") or "development").strip().lower()
    from backend.services.auth import is_demo_auth_allowed

    demo_auth_allowed = is_demo_auth_allowed()
    return {
        "status": "ready",
        "service": "nlcare_monitoring_prototype",
        "database": "ok",
        "environment": environment,
        "demo_auth_allowed": demo_auth_allowed,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "claim_boundary": (
            "Readiness means the engineering service can answer probes. It is "
            "not clinical validation, real-patient approval, or PHI compliance."
        ),
    }


# ─── Static files ─────────────────────────────────────────────────────────────

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
