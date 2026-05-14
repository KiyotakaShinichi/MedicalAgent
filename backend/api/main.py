"""
AI Breast Cancer Monitoring System — FastAPI application entrypoint.

This file wires together routers, middleware, static file mounts, and the
health-check / redirect routes. All business logic lives in routers/.
"""

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
from backend.api.routers.patient import router as patient_router
from backend.api.routers.clinician_review import router as clinician_review_router
from backend.api.routers.admin import router as admin_router
from backend.api.routers.model import router as model_router
from backend.api.routers.admin_eval import build_admin_eval_router
from backend.services.request_context import reset_request_id, set_request_id


# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="AI Breast Cancer Monitoring System")
ensure_schema()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid4())
    token = set_request_id(request_id)
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    reset_request_id(token)
    return response


# ─── Routers ──────────────────────────────────────────────────────────────────

app.include_router(auth_router)
app.include_router(patient_router)
app.include_router(clinician_review_router)
app.include_router(admin_router)
app.include_router(model_router)
app.include_router(build_admin_eval_router(get_admin_access_context, get_db))


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
        "service": "ai_breast_cancer_monitoring",
        "database": "ok",
    }


# ─── Static files ─────────────────────────────────────────────────────────────

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
app.mount("/artifacts", StaticFiles(directory="Data"), name="artifacts")
