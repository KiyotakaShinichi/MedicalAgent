from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.models import AccessSession, Base, Patient, PatientUpload
from backend.services import patient_uploads
from backend.services.auth import create_demo_session, get_context_from_authorization, revoke_session
from backend.services.rag_evidence_envelope import _high_risk_semantic_validation_required
from backend.services.synthetic_data_boundary import assert_synthetic_patient_id
from backend.services.upload_security import (
    UploadSecurityPolicy,
    decode_upload_payload,
    inspect_quarantined_upload,
    load_upload_security_policy,
)


def _db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_demo_session_persists_only_a_digest(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    db = _db()
    db.add(Patient(id="TEST-P001", name="Synthetic", diagnosis="Synthetic fixture"))
    db.commit()
    issued = create_demo_session(db, "patient", "TEST-P001")
    stored = db.query(AccessSession).one()
    assert stored.token.startswith("sha256$")
    assert issued["access_token"] not in stored.token
    assert get_context_from_authorization(db, f"Bearer {issued['access_token']}").patient_id == "TEST-P001"
    assert revoke_session(db, issued["access_token"]) is True


def test_strict_base64_rejects_malformed_payload():
    with pytest.raises(ValueError, match="strict base64"):
        decode_upload_payload("not base64!!!")


def test_upload_policy_defaults_disabled_in_staging():
    policy = load_upload_security_policy({"ENVIRONMENT": "staging"})
    assert policy.enabled is False
    assert policy.strict_profile is True


def test_staging_rejects_builtin_scanner():
    with pytest.raises(ValueError, match="external scanner"):
        load_upload_security_policy({
            "ENVIRONMENT": "staging",
            "NLCARE_UPLOADS_ENABLED": "true",
            "NLCARE_UPLOAD_SCANNER_MODE": "builtin",
        })


def test_magic_type_mismatch_is_rejected(tmp_path):
    path = tmp_path / "candidate"
    path.write_bytes(b"%PDF-1.7\n")
    with pytest.raises(ValueError, match="Declared content type"):
        inspect_quarantined_upload(
            path,
            file_name="report.pdf",
            declared_content_type="image/png",
            policy=UploadSecurityPolicy(True, False, "builtin", "", 10),
        )


def test_active_pdf_is_rejected(tmp_path):
    path = tmp_path / "candidate"
    path.write_bytes(b"%PDF-1.7\n1 0 obj <</JavaScript(test)>>")
    with pytest.raises(ValueError, match="Active or embedded"):
        inspect_quarantined_upload(
            path,
            file_name="report.pdf",
            declared_content_type="application/pdf",
            policy=UploadSecurityPolicy(True, False, "builtin", "", 10),
        )


def test_external_scanner_failure_keeps_file_quarantined(tmp_path, monkeypatch):
    script = tmp_path / "reject.py"
    script.write_text("raise SystemExit(2)\n", encoding="utf-8")
    quarantine = tmp_path / "quarantine"
    uploads = tmp_path / "uploads"
    monkeypatch.setattr(patient_uploads, "UPLOAD_QUARANTINE_DIR", quarantine)
    monkeypatch.setattr(patient_uploads, "UPLOAD_DIR", uploads)
    db = _db()
    db.add(Patient(id="TEST-P002", name="Synthetic", diagnosis="Synthetic fixture"))
    db.commit()
    policy = UploadSecurityPolicy(
        True,
        True,
        "external",
        f'"{sys.executable}" "{script}"',
        10,
    )
    payload = base64.b64encode(b"synthetic report").decode("ascii")
    with pytest.raises(ValueError, match="scanner rejected"):
        patient_uploads.save_patient_upload(
            db,
            "TEST-P002",
            "lab_report",
            "report.txt",
            "text/plain",
            payload,
            security_policy=policy,
        )
    assert db.query(PatientUpload).count() == 0
    assert list(quarantine.glob("*.blocked"))
    assert not list(uploads.rglob("*.txt"))


def test_successful_upload_records_security_manifest(tmp_path, monkeypatch):
    quarantine = tmp_path / "quarantine"
    uploads = tmp_path / "uploads"
    monkeypatch.setattr(patient_uploads, "UPLOAD_QUARANTINE_DIR", quarantine)
    monkeypatch.setattr(patient_uploads, "UPLOAD_DIR", uploads)
    db = _db()
    db.add(Patient(id="TEST-P003", name="Synthetic", diagnosis="Synthetic fixture"))
    db.commit()
    result = patient_uploads.save_patient_upload(
        db,
        "TEST-P003",
        "lab_report",
        "cbc.txt",
        "text/plain",
        base64.b64encode(b"WBC 5.1").decode("ascii"),
        security_policy=UploadSecurityPolicy(True, False, "builtin", "", 10),
    )
    assert result["upload_security"]["scanner_status"] == "builtin_passed"
    assert result["upload_security"]["sha256"]
    assert not list(quarantine.glob("*.pending"))


@pytest.mark.parametrize("patient_id", ["P001", "TEST-P100", "SYN-case-1", "DEMO_case"])
def test_synthetic_namespaces_are_accepted(monkeypatch, patient_id):
    monkeypatch.setenv("NLCARE_SYNTHETIC_ONLY", "true")
    assert_synthetic_patient_id(patient_id)


def test_non_synthetic_namespace_is_rejected(monkeypatch):
    monkeypatch.setenv("NLCARE_SYNTHETIC_ONLY", "true")
    with pytest.raises(PermissionError, match="non-synthetic"):
        assert_synthetic_patient_id("hospital-mrn-123")


def test_strict_high_risk_claim_requires_nli(monkeypatch):
    monkeypatch.setenv("NLCARE_HIGH_RISK_SEMANTIC_VALIDATION_REQUIRED", "true")
    assert _high_risk_semantic_validation_required([
        {"claim_type": "tumor_marker", "validation_method": "heuristic_overlap"}
    ]) is True
    assert _high_risk_semantic_validation_required([
        {"claim_type": "tumor_marker", "validation_method": "nli_entailment"}
    ]) is False


def test_disposable_staging_uses_leased_worker_and_synthetic_lock():
    compose = Path("docker-compose.synthetic-staging.yml").read_text(encoding="utf-8")
    assert "scripts/run_task_worker.py" not in compose
    assert "scripts/run_automation_worker.py" in compose
    assert "NLCARE_DATA_CLASSIFICATION: synthetic" in compose
    assert 'NLCARE_UPLOADS_ENABLED: "false"' in compose
