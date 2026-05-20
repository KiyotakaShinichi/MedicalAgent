"""Tests for the /admin/intent-classifier-probe endpoint.

Lock-ins:

  - GET / POST without auth -> 401/403
  - Patient token -> 401/403
  - Admin token + empty message -> ``status="empty"`` with the empty
    envelope; no exception
  - Admin token + real message -> ``status="ok"`` with deterministic +
    merged envelopes and an llm verdict block (available=False here
    because the test process has FAST_MODE=1)
  - ``use_llm=False`` short-circuits the LLM (llm.reason ==
    "use_llm_false") and merged == deterministic
"""
from __future__ import annotations

import os
import unittest

from fastapi.testclient import TestClient

from backend.api.main import app
from backend.services import local_llm


client = TestClient(app, raise_server_exceptions=False)


def _admin_token() -> str:
    resp = client.post(
        "/auth/demo-credential-login",
        json={"username": "admin", "password": "admin-demo"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def _patient_token() -> str:
    resp = client.post(
        "/auth/demo-credential-login",
        json={"username": "P001", "password": "P001"},
    )
    if resp.status_code != 200:
        resp = client.post(
            "/auth/demo-credential-login",
            json={"username": "patient", "password": "patient-demo"},
        )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


class IntentProbeAuth(unittest.TestCase):
    def test_unauthenticated_post_is_rejected(self) -> None:
        resp = client.post("/admin/intent-classifier-probe", json={"message": "hi"})
        self.assertIn(resp.status_code, {401, 403})

    def test_patient_token_is_rejected(self) -> None:
        token = _patient_token()
        resp = client.post(
            "/admin/intent-classifier-probe",
            headers=_auth(token),
            json={"message": "hi"},
        )
        self.assertIn(resp.status_code, {401, 403})


class IntentProbeBehavior(unittest.TestCase):
    def setUp(self) -> None:
        # Force LLM off so the test is hermetic (no Groq call).
        self._original_fm = os.environ.get("ONCOTRACK_FAST_MODE")
        os.environ["ONCOTRACK_FAST_MODE"] = "1"
        local_llm.set_fast_mode_override(None)
        self.token = _admin_token()

    def tearDown(self) -> None:
        local_llm.set_fast_mode_override(None)
        if self._original_fm is None:
            os.environ.pop("ONCOTRACK_FAST_MODE", None)
        else:
            os.environ["ONCOTRACK_FAST_MODE"] = self._original_fm

    def test_empty_message_returns_empty_status(self) -> None:
        resp = client.post(
            "/admin/intent-classifier-probe",
            headers=_auth(self.token),
            json={"message": ""},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "empty")
        self.assertIn("deterministic", body)
        self.assertIn("merged", body)
        self.assertFalse(body["llm"]["available"])

    def test_real_message_returns_envelopes(self) -> None:
        resp = client.post(
            "/admin/intent-classifier-probe",
            headers=_auth(self.token),
            json={"message": "hi, can you log my symptoms?"},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "ok")
        # Deterministic envelope catches greeting + tool request.
        det = body["deterministic"]
        self.assertTrue(det["has_casual_opener"])
        self.assertTrue(det["has_tool_request"])
        self.assertEqual(det["primary_intent"], "data_entry_intention")
        # Merged envelope is at least as informative.
        merged = body["merged"]
        self.assertEqual(merged["primary_intent"], "data_entry_intention")
        # LLM is off in this test (FAST_MODE=1), so llm.available is False.
        self.assertFalse(body["llm"]["available"])

    def test_use_llm_false_short_circuits(self) -> None:
        resp = client.post(
            "/admin/intent-classifier-probe",
            headers=_auth(self.token),
            json={"message": "hi", "use_llm": False},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["llm"]["reason"], "use_llm_false")
        # Merged is identical to deterministic when LLM is off.
        self.assertEqual(body["merged"], body["deterministic"])


if __name__ == "__main__":
    unittest.main()
