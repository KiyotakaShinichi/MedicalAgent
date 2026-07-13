from __future__ import annotations

from fastapi.testclient import TestClient

from backend.api.main import app


client = TestClient(app)


def _token(username: str, password: str) -> str:
    response = client.post(
        "/auth/demo-credential-login",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_automation_api_requires_admin():
    assert client.get("/admin/automation/capabilities").status_code in {401, 403}
    patient = _token("P001", "patient-demo")
    assert client.get("/admin/automation/capabilities", headers=_auth(patient)).status_code == 403


def test_admin_can_queue_and_run_dry_run_job():
    admin = _token("admin", "admin-demo")
    response = client.post(
        "/admin/automation/jobs",
        headers=_auth(admin),
        json={
            "job_type": "refresh_trace_envelope_v2_eval",
            "payload": {"run_id": "api-test", "reason": "contract test"},
            "dry_run": True,
            "run_in_background": True,
        },
    )
    assert response.status_code == 202, response.text
    task_id = response.json()["task"]["id"]
    task = client.get(f"/admin/automation/jobs/{task_id}", headers=_auth(admin))
    assert task.status_code == 200
    assert task.json()["status"] == "completed"
    assert task.json()["result"]["commands_executed"] is False
    assert task.json()["clinical_validation"] is False


def test_admin_api_rejects_nested_phi_payload():
    admin = _token("admin", "admin-demo")
    response = client.post(
        "/admin/automation/jobs",
        headers=_auth(admin),
        json={
            "job_type": "publish_trace_quality_digest",
            "payload": {"metadata": {"raw_patient_message": "blocked"}},
        },
    )
    assert response.status_code == 422
    assert "blocked_payload_fields_present" in response.text


def test_admin_capabilities_expose_only_engineering_posture():
    admin = _token("admin", "admin-demo")
    response = client.get("/admin/automation/capabilities", headers=_auth(admin))
    assert response.status_code == 200
    payload = response.json()
    assert payload["default_mode"] == "dry_run"
    assert payload["phi_allowed"] is False
    assert payload["clinical_validation"] is False
    assert "diagnosis" in payload["blocked_job_types"]
