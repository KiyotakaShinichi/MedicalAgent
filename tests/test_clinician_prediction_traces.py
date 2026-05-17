"""Endpoint test for the clinician-scoped prediction-trace view.

Confirms:
  - Only clinician + admin tokens can read the endpoint (patients are blocked).
  - 404 is returned for an unknown patient_id.
  - Filters (limit, abstained_only) are honored.
  - The response contains both per-patient and cohort summaries.
"""
from __future__ import annotations

import json
import unittest

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.api.main import app, get_db
from backend.database import Base
from backend.models import Patient, PredictionTrace


TEST_DB_URL = "sqlite:///:memory:"
engine = create_engine(
    TEST_DB_URL, connect_args={"check_same_thread": False}, poolclass=StaticPool,
)
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


def _seed():
    db = TestingSession()
    try:
        if not db.query(Patient).filter(Patient.id == "P001").first():
            db.add(Patient(id="P001", name="Trace Patient"))
            db.commit()
        # Seed a few traces for this patient.
        for i in range(3):
            db.add(PredictionTrace(
                patient_id="P001",
                actor_role="patient",
                question="response_classification",
                decision="insufficient_evidence" if i == 0 else "favorable_pattern",
                probability=None if i == 0 else 0.8,
                raw_probability=None if i == 0 else 0.85,
                calibrated=0 if i == 0 else 1,
                confidence="low" if i == 0 else "high",
                evidence_sufficiency="insufficient" if i == 0 else "sufficient",
                abstained=1 if i == 0 else 0,
                abstain_reason="no_response_signal_imaging_or_longitudinal_cbc_required" if i == 0 else None,
                modalities_present_json=json.dumps(["demographics"] if i == 0 else
                                                   ["demographics", "imaging", "cbc_pre", "cbc_nadir", "cbc_recovery"]),
                modalities_missing_json=json.dumps([] if i > 0 else
                                                   ["cbc_pre", "cbc_nadir", "cbc_recovery", "imaging", "symptoms", "interventions"]),
                confidence_modifier=0.0 if i == 0 else 1.0,
                model_version="gradient_boosting_treatment_success_binary",
                feature_set_version="synthetic_v2_2026_05",
                threshold_config_json=json.dumps({"lower": 0.4, "upper": 0.6}),
                calibration_config_json=json.dumps({"applied": i > 0}),
                safety_triggers_json=json.dumps([]),
                validator_decision="allowed",
                rag_source_ids_json=json.dumps([]),
            ))
        db.commit()
    finally:
        db.close()


_seed()


def override_get_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


client = TestClient(app, raise_server_exceptions=False)


def _login(username: str, password: str) -> str:
    resp = client.post(
        "/auth/demo-credential-login",
        json={"username": username, "password": password},
    )
    return resp.json().get("access_token")


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


class _DbOverrideMixin(unittest.TestCase):
    """Re-pin the FastAPI `get_db` override before every test method.

    pytest runs multiple test modules in the same process, and any module
    that also overrides `get_db` (e.g. tests/test_access_control.py) will
    win whichever is loaded last.  Re-pinning in `setUp` makes each test
    deterministic regardless of load order.
    """

    def setUp(self) -> None:
        app.dependency_overrides[get_db] = override_get_db


class ClinicianTracesAccessControl(_DbOverrideMixin):
    def test_unauthenticated_request_is_rejected(self) -> None:
        resp = client.get("/clinician/patients/P001/prediction-traces")
        self.assertEqual(resp.status_code, 401)

    def test_patient_token_is_blocked(self) -> None:
        token = _login("P001", "patient-demo")
        self.assertIsNotNone(token)
        resp = client.get(
            "/clinician/patients/P001/prediction-traces",
            headers=_auth(token),
        )
        self.assertIn(resp.status_code, (401, 403))

    def test_clinician_token_can_read(self) -> None:
        token = _login("clinician", "clinician-demo")
        self.assertIsNotNone(token)
        resp = client.get(
            "/clinician/patients/P001/prediction-traces",
            headers=_auth(token),
        )
        self.assertEqual(resp.status_code, 200, resp.text)

    def test_admin_token_can_read(self) -> None:
        token = _login("admin", "admin-demo")
        self.assertIsNotNone(token)
        resp = client.get(
            "/clinician/patients/P001/prediction-traces",
            headers=_auth(token),
        )
        self.assertEqual(resp.status_code, 200, resp.text)


class ClinicianTracesContract(_DbOverrideMixin):
    def setUp(self) -> None:
        super().setUp()
        self.token = _login("clinician", "clinician-demo")

    def test_response_has_required_top_level_keys(self) -> None:
        resp = client.get(
            "/clinician/patients/P001/prediction-traces",
            headers=_auth(self.token),
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        for key in ("patient_id", "traces", "patient_summary", "cohort_summary", "claim_boundary"):
            self.assertIn(key, payload)
        self.assertEqual(payload["patient_id"], "P001")
        self.assertGreaterEqual(len(payload["traces"]), 3)

    def test_abstained_only_filter_works(self) -> None:
        resp = client.get(
            "/clinician/patients/P001/prediction-traces?abstained_only=true",
            headers=_auth(self.token),
        )
        self.assertEqual(resp.status_code, 200)
        traces = resp.json()["traces"]
        self.assertEqual(len(traces), 1)
        self.assertTrue(traces[0]["abstained"])

    def test_limit_is_honored(self) -> None:
        resp = client.get(
            "/clinician/patients/P001/prediction-traces?limit=2",
            headers=_auth(self.token),
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()["traces"]), 2)

    def test_unknown_patient_returns_404(self) -> None:
        resp = client.get(
            "/clinician/patients/UNKNOWN/prediction-traces",
            headers=_auth(self.token),
        )
        self.assertEqual(resp.status_code, 404)

    def test_patient_summary_counts_match_seeded_data(self) -> None:
        resp = client.get(
            "/clinician/patients/P001/prediction-traces",
            headers=_auth(self.token),
        )
        payload = resp.json()
        summary = payload["patient_summary"]
        self.assertGreaterEqual(summary["total"], 3)
        # One of the three seeded traces is an abstention.
        self.assertGreater(summary["abstention_rate"] or 0, 0)
        # Decision breakdown includes both decisions.
        self.assertIn("insufficient_evidence", summary["decision_counts"])
        self.assertIn("favorable_pattern", summary["decision_counts"])


if __name__ == "__main__":
    unittest.main()
