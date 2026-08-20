"""Tests for the admin FAST_MODE toggle endpoints.

Locks in:

  - GET ``/admin/fast-mode`` returns the resolved state + source.
  - POST ``/admin/fast-mode`` flips the runtime override (True / False
    / None) and the new state is observable via the GET endpoint.
  - The endpoints require admin auth (a patient token is rejected).
  - Toggling actually affects the adjudicator: setting True makes
    ``_adjudicate_json`` short-circuit to ``available=False``.
"""
from __future__ import annotations

import os
import unittest

from fastapi.testclient import TestClient

from backend.api.main import app
from backend.database import SessionLocal
from backend.models import Patient
from backend.services import local_llm


client = TestClient(app, raise_server_exceptions=False)


def _ensure_synthetic_patient() -> None:
    db = SessionLocal()
    try:
        if db.query(Patient).filter(Patient.id == "P001").first() is None:
            db.add(Patient(id="P001", name="Synthetic auth fixture", diagnosis="Synthetic test data"))
            db.commit()
    finally:
        db.close()


def _admin_token() -> str:
    resp = client.post(
        "/auth/demo-credential-login",
        json={"username": "admin", "password": "admin-demo"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def _patient_token() -> str:
    _ensure_synthetic_patient()
    resp = client.post(
        "/auth/demo-credential-login",
        json={"username": "P001", "password": "P001"},
    )
    if resp.status_code != 200:
        # Fallback: some seeds use a different demo password.
        resp = client.post(
            "/auth/demo-credential-login",
            json={"username": "patient", "password": "patient-demo"},
        )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


class FastModeEndpointAuth(unittest.TestCase):
    def setUp(self) -> None:
        local_llm.set_fast_mode_override(None)
        self._original_env = os.environ.get("ONCOTRACK_FAST_MODE")
        os.environ.pop("ONCOTRACK_FAST_MODE", None)

    def tearDown(self) -> None:
        local_llm.set_fast_mode_override(None)
        if self._original_env is None:
            os.environ.pop("ONCOTRACK_FAST_MODE", None)
        else:
            os.environ["ONCOTRACK_FAST_MODE"] = self._original_env

    def test_unauthenticated_get_is_rejected(self) -> None:
        resp = client.get("/admin/fast-mode")
        self.assertIn(resp.status_code, {401, 403})

    def test_unauthenticated_post_is_rejected(self) -> None:
        resp = client.post("/admin/fast-mode", json={"enabled": True})
        self.assertIn(resp.status_code, {401, 403})

    def test_patient_token_is_rejected(self) -> None:
        token = _patient_token()
        resp = client.get("/admin/fast-mode", headers=_auth(token))
        self.assertIn(resp.status_code, {401, 403})


class FastModeEndpointBehavior(unittest.TestCase):
    def setUp(self) -> None:
        local_llm.set_fast_mode_override(None)
        self._original_env = os.environ.get("ONCOTRACK_FAST_MODE")
        os.environ.pop("ONCOTRACK_FAST_MODE", None)
        self.token = _admin_token()

    def tearDown(self) -> None:
        local_llm.set_fast_mode_override(None)
        if self._original_env is None:
            os.environ.pop("ONCOTRACK_FAST_MODE", None)
        else:
            os.environ["ONCOTRACK_FAST_MODE"] = self._original_env

    def test_admin_can_read_default_state(self) -> None:
        resp = client.get("/admin/fast-mode", headers=_auth(self.token))
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertFalse(payload["enabled"])
        self.assertEqual(payload["source"], "env_var")
        self.assertIsNone(payload["runtime_override"])

    def test_admin_can_enable_runtime_override(self) -> None:
        resp = client.post(
            "/admin/fast-mode",
            headers=_auth(self.token),
            json={"enabled": True},
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["source"], "runtime_override")
        # And the adjudicator now short-circuits.
        result = local_llm._adjudicate_json(system="x", prompt="y")
        self.assertFalse(result["available"])

    def test_admin_can_clear_runtime_override(self) -> None:
        # Force ON then clear.
        client.post("/admin/fast-mode", headers=_auth(self.token), json={"enabled": True})
        resp = client.post(
            "/admin/fast-mode",
            headers=_auth(self.token),
            json={"enabled": None},
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertFalse(payload["enabled"])
        self.assertEqual(payload["source"], "env_var")
        self.assertIsNone(payload["runtime_override"])

    def test_admin_can_explicitly_disable(self) -> None:
        # Env var ON, runtime override OFF -> off wins.
        os.environ["ONCOTRACK_FAST_MODE"] = "1"
        resp = client.post(
            "/admin/fast-mode",
            headers=_auth(self.token),
            json={"enabled": False},
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertFalse(payload["enabled"])
        self.assertEqual(payload["source"], "runtime_override")
        self.assertFalse(payload["runtime_override"])


if __name__ == "__main__":
    unittest.main()
