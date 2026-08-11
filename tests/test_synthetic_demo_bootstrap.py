from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base
from backend.models import LabResult, Patient, SymptomReport
from backend.services.auth import create_demo_session_from_credentials
from backend.services.synthetic_demo_bootstrap import (
    ensure_synthetic_demo_data,
    synthetic_demo_bootstrap_allowed,
)


def _session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_bootstrap_creates_both_scoped_demo_accounts() -> None:
    db = _session()
    try:
        result = ensure_synthetic_demo_data(db)
        assert result["created_patient_ids"] == ["P001", "P002"]
        assert {row.id for row in db.query(Patient).all()} == {"P001", "P002"}
        assert db.query(LabResult).filter(LabResult.patient_id == "P001").count() > 0
        assert db.query(LabResult).filter(LabResult.patient_id == "P002").count() == 0

        p001 = create_demo_session_from_credentials(db, "P001", "patient-demo")
        p002 = create_demo_session_from_credentials(db, "P002", "patient-demo")
        assert p001["patient_id"] == "P001"
        assert p002["patient_id"] == "P002"
    finally:
        db.close()


def test_bootstrap_is_idempotent_and_preserves_existing_records() -> None:
    db = _session()
    try:
        ensure_synthetic_demo_data(db)
        db.add(
            SymptomReport(
                patient_id="P001",
                date=date(2026, 1, 1),
                symptom="synthetic test note",
                severity=1,
                notes="Regression fixture only.",
            )
        )
        db.commit()
        before = db.query(SymptomReport).filter(SymptomReport.patient_id == "P001").count()

        result = ensure_synthetic_demo_data(db)
        after = db.query(SymptomReport).filter(SymptomReport.patient_id == "P001").count()

        assert result["created_patient_ids"] == []
        assert set(result["existing_patient_ids"]) == {"P001", "P002"}
        assert after == before
    finally:
        db.close()


def test_bootstrap_posture_fails_closed_outside_synthetic_profile() -> None:
    allowed = {
        "APP_ENV": "synthetic_staging",
        "NLCARE_SYNTHETIC_ONLY": "true",
        "NLCARE_DATA_CLASSIFICATION": "synthetic",
        "NLCARE_BOOTSTRAP_SYNTHETIC_DEMO": "true",
    }
    assert synthetic_demo_bootstrap_allowed(allowed) is True
    assert synthetic_demo_bootstrap_allowed({**allowed, "APP_ENV": "production"}) is False
    assert synthetic_demo_bootstrap_allowed({**allowed, "NLCARE_SYNTHETIC_ONLY": "false"}) is False
    assert synthetic_demo_bootstrap_allowed({**allowed, "NLCARE_DATA_CLASSIFICATION": "phi"}) is False
