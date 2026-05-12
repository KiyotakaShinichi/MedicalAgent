"""
Auth router — credential login, demo patients list, whoami.

Extracted from main.py as part of the router-split refactor.
To wire into main.py:
    from backend.api.routers.auth import router as auth_router
    app.include_router(auth_router)
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.api.deps import get_db, get_access_context
from backend.api.response_models import (
    LoginResponse,
    DemoPatientsResponse,
    WhoAmIResponse,
)
from backend.crud import get_all_patients

router = APIRouter(prefix="/auth", tags=["auth"])


class DemoLoginRequest(BaseModel):
    role: str
    patient_id: str | None = None


class DemoCredentialLoginRequest(BaseModel):
    username: str
    password: str


@router.post("/demo-login", response_model=LoginResponse)
def demo_login(payload: DemoLoginRequest, db: Session = Depends(get_db)):
    """Legacy role-selector login (kept for backwards compat with HTML frontend)."""
    from backend.services.auth import create_demo_session
    try:
        return create_demo_session(db, role=payload.role, patient_id=payload.patient_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/demo-credential-login", response_model=LoginResponse)
def demo_credential_login(payload: DemoCredentialLoginRequest, db: Session = Depends(get_db)):
    """
    Credential-based demo login.
    Role is inferred from username — no client-side role selection.
    Patients are scoped to their own records only.
    """
    from backend.services.auth import create_demo_session_from_credentials
    try:
        return create_demo_session_from_credentials(db, username=payload.username, password=payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.get("/demo-patients", response_model=DemoPatientsResponse)
def demo_patient_options(db: Session = Depends(get_db)):
    """List demo patient IDs. Used by the legacy HTML login form only."""
    patients = get_all_patients(db)[:50]
    return {
        "patients": [
            {"id": p.id, "label": f"Demo patient {i}", "hint": p.id}
            for i, p in enumerate(patients, start=1)
        ]
    }


@router.get("/whoami", response_model=WhoAmIResponse)
def whoami(context=Depends(get_access_context)):
    """Return the role and patient_id for the current bearer token."""
    return {"role": context.role, "patient_id": context.patient_id}
