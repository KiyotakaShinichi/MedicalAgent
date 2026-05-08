from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import secrets

from backend.models import AccessSession, Patient, UserAccount


VALID_ROLES = {"patient", "clinician", "admin"}
DEMO_ROLE_CREDENTIALS = {
    "admin": {
        "aliases": {"admin", "admin@demo.local", "demo-admin"},
        "password": "admin-demo",
    },
    "clinician": {
        "aliases": {"clinician", "clinician@demo.local", "demo-clinician"},
        "password": "clinician-demo",
    },
}


@dataclass
class AccessContext:
    role: str
    patient_id: str | None
    token: str


def create_demo_session(db, role: str, patient_id: str | None = None):
    normalized_role = role.lower().strip()
    if normalized_role not in VALID_ROLES:
        raise ValueError("role must be patient, clinician, or admin")
    if normalized_role == "patient":
        if not patient_id:
            raise ValueError("patient_id is required for patient sessions")
        if db.query(Patient).filter(Patient.id == patient_id).first() is None:
            raise ValueError("patient not found")

    _ensure_demo_account(db, normalized_role, patient_id)
    session = AccessSession(
        token=secrets.token_urlsafe(32),
        role=normalized_role,
        patient_id=patient_id if normalized_role == "patient" else None,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=12),
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    return {
        "access_token": session.token,
        "token_type": "bearer",
        "role": session.role,
        "patient_id": session.patient_id,
        "expires_at": session.expires_at.isoformat(),
        "warning": "Demo session only. Replace with real authentication before deployment.",
    }


def create_demo_session_from_credentials(db, username: str, password: str):
    normalized_username = (username or "").strip().lower()
    normalized_password = (password or "").strip()
    if not normalized_username or not normalized_password:
        raise ValueError("username and password are required")

    for role, credential in DEMO_ROLE_CREDENTIALS.items():
        if (
            normalized_username in credential["aliases"]
            and normalized_password == credential["password"]
        ):
            return create_demo_session(db, role=role)

    patient = _patient_from_demo_username(db, normalized_username)
    if patient is not None and normalized_password in {"patient-demo", normalized_username, patient.id}:
        return create_demo_session(db, role="patient", patient_id=patient.id)

    raise ValueError("Invalid demo credentials")


def get_context_from_authorization(db, authorization_header: str | None):
    if not authorization_header or not authorization_header.lower().startswith("bearer "):
        raise PermissionError("Missing bearer token")

    token = authorization_header.split(" ", 1)[1].strip()
    session = db.query(AccessSession).filter(AccessSession.token == token).first()
    if session is None:
        raise PermissionError("Invalid bearer token")

    expires_at = session.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < datetime.now(timezone.utc):
        raise PermissionError("Expired bearer token")

    return AccessContext(role=session.role, patient_id=session.patient_id, token=token)


def require_patient_context(context: AccessContext):
    if context.role != "patient" or not context.patient_id:
        raise PermissionError("Patient session required")
    return context


def require_admin_or_clinician(context: AccessContext):
    if context.role not in {"admin", "clinician"}:
        raise PermissionError("Clinician or admin session required")
    return context


def require_admin_context(context: AccessContext):
    if context.role != "admin":
        raise PermissionError("Admin session required")
    return context


def _ensure_demo_account(db, role, patient_id):
    username = f"demo-{role}-{patient_id or 'global'}"
    if db.query(UserAccount).filter(UserAccount.username == username).first():
        return
    db.add(UserAccount(
        username=username,
        role=role,
        patient_id=patient_id if role == "patient" else None,
        display_name=f"Demo {role.title()}",
    ))


def _patient_from_demo_username(db, normalized_username):
    if not normalized_username:
        return None
    exact = db.query(Patient).filter(Patient.id == normalized_username).first()
    if exact is not None:
        return exact
    for patient in db.query(Patient).all():
        if patient.id.lower() == normalized_username:
            return patient
    return None
