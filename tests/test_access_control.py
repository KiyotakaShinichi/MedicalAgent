"""
Access-control and role-boundary tests.

These tests verify that:
- Patients cannot access other patients' records
- Patients cannot reach admin or clinician endpoints
- Clinicians cannot reach admin-only endpoints
- Unauthenticated requests are rejected with 401
- Admin tokens are accepted on admin-only routes

All tests use the FastAPI TestClient with an in-memory SQLite database
seeded with demo credentials via the same auth service used in production.
"""

import unittest
from unittest.mock import patch
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base
from backend.models import Patient, PatientUpload
from backend.config import UPLOAD_DIR
from backend.api.main import app, get_db

# ── In-memory DB for tests ────────────────────────────────────────────────────
# StaticPool ensures all sessions share the same in-memory SQLite connection,
# so tables created by create_all are visible to every subsequent session.

TEST_DB_URL = "sqlite:///:memory:"

engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def _seed():
    db = TestingSession()
    try:
        if not db.query(Patient).filter(Patient.id == "P001").first():
            db.add(Patient(id="P001", name="Demo Patient 1"))
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


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app, raise_server_exceptions=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_token(username: str, password: str) -> str | None:
    resp = client.post("/auth/demo-credential-login", json={"username": username, "password": password})
    if resp.status_code == 200:
        return resp.json().get("access_token")
    return None


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── Auth endpoint tests ───────────────────────────────────────────────────────

class TestAuthEndpoints(unittest.TestCase):

    def test_credential_login_patient(self):
        resp = client.post("/auth/demo-credential-login", json={"username": "P001", "password": "patient-demo"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["role"], "patient")
        self.assertIsNotNone(data.get("access_token"))
        self.assertEqual(data.get("patient_id"), "P001")

    def test_credential_login_clinician(self):
        resp = client.post("/auth/demo-credential-login", json={"username": "clinician", "password": "clinician-demo"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["role"], "clinician")

    def test_credential_login_admin(self):
        resp = client.post("/auth/demo-credential-login", json={"username": "admin", "password": "admin-demo"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["role"], "admin")

    def test_invalid_credentials_rejected(self):
        resp = client.post("/auth/demo-credential-login", json={"username": "hacker", "password": "wrong"})
        self.assertIn(resp.status_code, [401, 400])

    def test_whoami_unauthenticated(self):
        resp = client.get("/auth/whoami")
        self.assertIn(resp.status_code, [401, 403])

    def test_whoami_with_patient_token(self):
        token = _get_token("P001", "patient-demo")
        if not token:
            self.skipTest("Could not obtain P001 token")
        resp = client.get("/auth/whoami", headers=_auth(token))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["role"], "patient")

    def test_logout_revokes_server_session(self):
        token = _get_token("P001", "patient-demo")
        if not token:
            self.skipTest("Could not obtain P001 token")
        logout = client.delete("/auth/session", headers=_auth(token))
        self.assertEqual(logout.status_code, 200)
        self.assertTrue(logout.json()["revoked"])
        whoami = client.get("/auth/whoami", headers=_auth(token))
        self.assertIn(whoami.status_code, [401, 403])

    def test_raw_data_artifact_mount_is_not_public(self):
        response = client.get("/artifacts/evals/rag/latest_rag_baseline_comparison.json")
        self.assertEqual(response.status_code, 404)


# ── Patient isolation tests ───────────────────────────────────────────────────

class TestPatientIsolation(unittest.TestCase):

    def test_patient_can_read_own_report(self):
        token = _get_token("P001", "patient-demo")
        if not token:
            self.skipTest("Could not obtain P001 token")
        resp = client.get("/me/patient-report", headers=_auth(token))
        # 200 or 404 (no data seeded in test DB), but NOT 401/403
        self.assertNotIn(resp.status_code, [401, 403])

    def test_unauthenticated_cannot_read_patient_report(self):
        resp = client.get("/me/patient-report")
        self.assertIn(resp.status_code, [401, 403])

    def test_patient_cannot_read_admin_analytics(self):
        token = _get_token("P001", "patient-demo")
        if not token:
            self.skipTest("Could not obtain P001 token")
        resp = client.get("/admin/analytics", headers=_auth(token))
        self.assertIn(resp.status_code, [401, 403])

    def test_patient_cannot_trigger_agent_regression(self):
        token = _get_token("P001", "patient-demo")
        if not token:
            self.skipTest("Could not obtain P001 token")
        resp = client.post("/admin/agent-regression", headers=_auth(token))
        self.assertIn(resp.status_code, [401, 403])

    def test_patient_cannot_list_all_patients(self):
        token = _get_token("P001", "patient-demo")
        if not token:
            self.skipTest("Could not obtain P001 token")
        resp = client.get("/patients", headers=_auth(token))
        self.assertIn(resp.status_code, [401, 403])

    def test_patient_upload_content_requires_owner_token_and_hides_local_path(self):
        token = _get_token("P001", "patient-demo")
        if not token:
            self.skipTest("Could not obtain P001 token")
        path = Path(UPLOAD_DIR) / "access-control" / "protected-test.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic protected upload", encoding="utf-8")
        db = TestingSession()
        try:
            upload = PatientUpload(
                patient_id="P001",
                upload_type="test",
                original_filename="protected-test.txt",
                content_type="text/plain",
                local_path=str(path),
                notes="synthetic access-control fixture",
            )
            db.add(upload)
            db.commit()
            db.refresh(upload)

            unauthenticated = client.get(f"/me/uploads/{upload.id}/content")
            self.assertIn(unauthenticated.status_code, [401, 403])
            owned = client.get(f"/me/uploads/{upload.id}/content", headers=_auth(token))
            self.assertEqual(owned.status_code, 200)
            self.assertEqual(owned.text, "synthetic protected upload")

            listing = client.get("/me/uploads", headers=_auth(token))
            self.assertEqual(listing.status_code, 200)
            matching = next(row for row in listing.json()["uploads"] if row["id"] == upload.id)
            self.assertNotIn("local_path", matching)
            self.assertEqual(matching["content_url"], f"/me/uploads/{upload.id}/content")
        finally:
            db.query(PatientUpload).filter(PatientUpload.local_path == str(path)).delete()
            db.commit()
            db.close()
            path.unlink(missing_ok=True)


# ── Clinician access tests ────────────────────────────────────────────────────

class TestClinicianAccess(unittest.TestCase):

    def test_clinician_can_read_patients(self):
        token = _get_token("clinician", "clinician-demo")
        if not token:
            self.skipTest("Could not obtain clinician token")
        resp = client.get("/patients", headers=_auth(token))
        self.assertNotIn(resp.status_code, [401, 403])

    def test_clinician_can_read_review_queue(self):
        token = _get_token("clinician", "clinician-demo")
        if not token:
            self.skipTest("Could not obtain clinician token")
        resp = client.get("/clinician/review-queue", headers=_auth(token))
        self.assertNotIn(resp.status_code, [401, 403])

    def test_clinician_cannot_access_admin_mle(self):
        token = _get_token("clinician", "clinician-demo")
        if not token:
            self.skipTest("Could not obtain clinician token")
        resp = client.post("/admin/mle-readiness", headers=_auth(token))
        self.assertIn(resp.status_code, [401, 403])

    def test_clinician_cannot_trigger_agent_regression(self):
        token = _get_token("clinician", "clinician-demo")
        if not token:
            self.skipTest("Could not obtain clinician token")
        resp = client.post("/admin/agent-regression", headers=_auth(token))
        self.assertIn(resp.status_code, [401, 403])


# ── Admin access tests ────────────────────────────────────────────────────────

class TestAdminAccess(unittest.TestCase):

    def test_admin_can_read_analytics(self):
        token = _get_token("admin", "admin-demo")
        if not token:
            self.skipTest("Could not obtain admin token")
        resp = client.get("/admin/analytics", headers=_auth(token))
        self.assertNotIn(resp.status_code, [401, 403])

    def test_admin_can_trigger_mle_readiness(self):
        token = _get_token("admin", "admin-demo")
        if not token:
            self.skipTest("Could not obtain admin token")
        resp = client.post("/admin/mle-readiness", headers=_auth(token))
        self.assertNotIn(resp.status_code, [401, 403])

    def test_unauthenticated_cannot_access_admin(self):
        resp = client.get("/admin/analytics")
        self.assertIn(resp.status_code, [401, 403])


# ── Health endpoint ───────────────────────────────────────────────────────────

class TestHealthEndpoint(unittest.TestCase):

    def test_health_unauthenticated(self):
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("status", resp.json())
        self.assertEqual(resp.headers.get("x-content-type-options"), "nosniff")
        self.assertEqual(resp.headers.get("x-frame-options"), "DENY")
        self.assertEqual(resp.headers.get("referrer-policy"), "no-referrer")
        self.assertIn("frame-ancestors 'none'", resp.headers.get("content-security-policy", ""))
        self.assertEqual(resp.headers.get("x-api-protection-scope"), "process_local_engineering_control")

    def test_ready_unauthenticated_with_boundary(self):
        from backend.services import rag_vector_index

        ready_state = {
            "status": "ready",
            "backend": "local_sparse_tfidf_bm25_index",
            "document_count": 1,
            "startup_warmup_ms": 1.0,
            "error_type": None,
            "clinical_validation": False,
            "healthcare_production_ready": False,
        }
        with patch.dict(
            rag_vector_index._PREWARM_STATE,
            ready_state,
            clear=True,
        ), patch.dict("os.environ", {"NLCARE_RAG_REQUIRE_DENSE": "false"}):
            resp = client.get("/ready")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ready")
        self.assertFalse(data["clinical_validation"])
        self.assertFalse(data["healthcare_production_ready"])
        self.assertIn("claim_boundary", data)


if __name__ == "__main__":
    unittest.main()
