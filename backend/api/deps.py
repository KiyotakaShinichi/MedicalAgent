"""
Shared FastAPI dependency functions for auth, DB session, and role-based access.

Import pattern:
    from backend.api.deps import get_db, get_patient_access_context, get_admin_access_context
"""

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from backend.database import SessionLocal


# ─── Database ─────────────────────────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─── Auth contexts ────────────────────────────────────────────────────────────

def get_access_context(
    authorization: str | None = Header(None),
    db: Session = Depends(get_db),
):
    """Resolve bearer token → role context. Returns None for unauthenticated."""
    from backend.services.auth import get_context_from_authorization
    try:
        return get_context_from_authorization(db, authorization)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def get_patient_access_context(context=Depends(get_access_context)):
    """Require patient role. Raises 403 if caller is not a patient."""
    from backend.services.auth import require_patient_context
    try:
        return require_patient_context(context)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def get_clinician_or_admin_context(context=Depends(get_access_context)):
    """Require clinician or admin role."""
    from backend.services.auth import require_admin_or_clinician
    try:
        return require_admin_or_clinician(context)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def get_admin_access_context(context=Depends(get_access_context)):
    """Require admin role. Raises 403 for patients and clinicians."""
    from backend.services.auth import require_admin_context
    try:
        return require_admin_context(context)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
