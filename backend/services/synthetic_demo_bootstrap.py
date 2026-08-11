"""Idempotent bootstrap for the local synthetic demonstration accounts."""

from __future__ import annotations

from collections.abc import Mapping

from backend.models import Patient
from backend.services.demo_patient_sync import sync_demo_patient_journey


DEMO_PATIENT_IDS = ("P001", "P002")


def synthetic_demo_bootstrap_allowed(environment: Mapping[str, str]) -> bool:
    """Fail closed unless the runtime is explicitly synthetic and nonproduction."""

    enabled = _truthy(environment.get("NLCARE_BOOTSTRAP_SYNTHETIC_DEMO"))
    synthetic_only = _truthy(environment.get("NLCARE_SYNTHETIC_ONLY"))
    data_class = str(environment.get("NLCARE_DATA_CLASSIFICATION") or "").strip().lower()
    profile = str(
        environment.get("ENVIRONMENT")
        or environment.get("APP_ENV")
        or "development"
    ).strip().lower()
    return (
        enabled
        and synthetic_only
        and data_class == "synthetic"
        and profile in {"development", "test", "synthetic_staging"}
    )


def ensure_synthetic_demo_data(db) -> dict:
    """Create missing demo accounts without deleting or rewriting existing data."""

    created: list[str] = []
    existing: list[str] = []

    patient_one = db.query(Patient).filter(Patient.id == "P001").first()
    if patient_one is None:
        sync_demo_patient_journey(db, patient_id="P001")
        created.append("P001")
    else:
        existing.append("P001")

    patient_two = db.query(Patient).filter(Patient.id == "P002").first()
    if patient_two is None:
        db.add(
            Patient(
                id="P002",
                name="Patient P002",
                diagnosis="Synthetic breast monitoring demo",
            )
        )
        db.commit()
        created.append("P002")
    else:
        existing.append("P002")

    return {
        "status": "ready",
        "created_patient_ids": created,
        "existing_patient_ids": existing,
        "patient_ids": list(DEMO_PATIENT_IDS),
        "synthetic_only": True,
        "clinical_validation": False,
    }


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}

