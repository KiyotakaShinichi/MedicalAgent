from __future__ import annotations

from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.routers import patient as patient_router


client = TestClient(app)


def _patient_token() -> str:
    response = client.post(
        "/auth/demo-credential-login",
        json={"username": "P001", "password": "patient-demo"},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def test_core_report_returns_records_with_deferred_enrichment(monkeypatch):
    monkeypatch.setattr(
        patient_router,
        "_schedule_report_enrichment",
        lambda patient_id: {"status": "queued", "retry_after_ms": 750},
    )
    token = _patient_token()
    response = client.get("/me/patient-report/core", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == "P001"
    assert data["report_enrichment"]["status"] == "deferred"
    assert data["report_enrichment"]["clinical_validation"] is False
    assert "latest_labs" in data
    assert data["hybrid_prediction"] is None


def test_enrichment_endpoint_returns_pending_without_blocking(monkeypatch):
    monkeypatch.setattr(patient_router, "_get_cached_report", lambda patient_id: None)
    monkeypatch.setattr(patient_router, "get_patient_enrichment_job", lambda patient_id: None)
    monkeypatch.setattr(
        patient_router,
        "_schedule_report_enrichment",
        lambda patient_id: {
            "status": "queued",
            "retry_after_ms": 750,
            "clinical_validation": False,
            "healthcare_production_ready": False,
        },
    )
    token = _patient_token()
    response = client.get("/me/patient-report/enrichment", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    data = response.json()
    assert data["report_enrichment"]["status"] == "queued"
    assert data["report_enrichment"]["retry_after_ms"] == 750
    assert data["report_enrichment"]["clinical_validation"] is False
    assert "hybrid_prediction" in data
    assert data["hybrid_prediction"] is None
    assert "latest_labs" not in data


def test_enrichment_endpoint_returns_cached_deferred_fields(monkeypatch):
    monkeypatch.setattr(
        patient_router,
        "_get_cached_report",
        lambda patient_id: {
            "hybrid_prediction": {"decision": "synthetic example"},
            "report_enrichment": {
                "status": "complete",
                "clinical_validation": False,
            },
        },
    )
    token = _patient_token()
    response = client.get("/me/patient-report/enrichment", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    data = response.json()
    assert data["report_enrichment"]["status"] == "complete"
    assert data["hybrid_prediction"]["decision"] == "synthetic example"
    assert "latest_labs" not in data
