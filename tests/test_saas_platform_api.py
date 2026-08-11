from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.api.main import app, get_db
from backend.database import Base
from backend.models import Patient


ENGINE = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSession = sessionmaker(bind=ENGINE, autocommit=False, autoflush=False)
Base.metadata.create_all(ENGINE)
with TestingSession() as seed_db:
    seed_db.add(Patient(id="P001", name="Synthetic Demo Patient"))
    seed_db.commit()


def _override_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def _pin_database_override(monkeypatch):
    monkeypatch.setenv("ALLOW_DEMO_AUTH", "true")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("NLCARE_SHARED_RATE_LIMIT_ENABLED", "false")
    previous = app.dependency_overrides.get(get_db)
    app.dependency_overrides[get_db] = _override_db
    try:
        yield
    finally:
        if previous is None:
            app.dependency_overrides.pop(get_db, None)
        else:
            app.dependency_overrides[get_db] = previous


def _token(client: TestClient, username: str, password: str) -> str:
    response = client.post(
        "/auth/demo-credential-login",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def test_admin_workspace_session_and_idempotent_job_api():
    client = TestClient(app, raise_server_exceptions=False)
    token = _token(client, "admin", "admin-demo")
    auth = {"Authorization": f"Bearer {token}"}
    session = client.get("/platform/session", headers=auth)
    assert session.status_code == 200
    body = session.json()
    assert body["clinical_validation"] is False
    assert body["billing_enabled"] is False
    assert len(body["organizations"]) == 1
    organization_id = body["organizations"][0]["id"]

    overview = client.get(f"/platform/organizations/{organization_id}/overview", headers=auth)
    assert overview.status_code == 200
    project_id = overview.json()["projects"][0]["id"]
    endpoint = f"/platform/organizations/{organization_id}/projects/{project_id}/jobs"
    payload = {"job_type": "release_gate", "payload": {"dry_run": True}}
    headers = auth | {"Idempotency-Key": "api-release-gate-0001"}
    first = client.post(endpoint, headers=headers, json=payload)
    second = client.post(endpoint, headers=headers, json=payload)
    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["job"]["id"] == second.json()["job"]["id"]
    assert first.json()["idempotent_reuse"] is False
    assert second.json()["idempotent_reuse"] is True


def test_patient_cannot_create_a_saas_organization():
    client = TestClient(app, raise_server_exceptions=False)
    token = _token(client, "P001", "patient-demo")
    response = client.post(
        "/platform/organizations",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "Patient Owned Workspace"},
    )
    assert response.status_code == 403
