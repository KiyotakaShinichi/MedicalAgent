"""Malformed request bodies are rejected with HTTP 422 at the API boundary.

Both write boundaries bind a Pydantic request model, so FastAPI rejects a body
that does not match before any handler code runs:

    POST /patients                    -> PatientCreate
    POST /patients/{patient_id}/chat  -> PatientChatRequest

Each route is exercised authenticated, because authentication runs first: an
unauthenticated malformed request returns 401 and proves nothing about
validation. Every assertion below is an explicit `== 422` against a real
response, with a valid-request control per route so the 422s are known to come
from the payload rather than from a permanently broken endpoint.

`tests/test_api_input_validation_boundaries.py` holds the deeper guards — that
the routes stay bound to typed models at all, that the OpenAPI document
publishes named request schemas, and that `/patient-report` remains a read
boundary. This file is the direct, readable evidence that malformed input is
refused.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.main import app  # noqa: E402

PATIENTS_URL = "/patients"
CHAT_URL = "/patients/P001/chat"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def auth_headers(client: TestClient) -> dict[str, str]:
    """A demo clinician token.

    Authentication is checked before request-body validation, so without this
    every malformed request would return 401 and the 422 contract would never
    be reached.
    """
    response = client.post("/auth/demo-login", json={"role": "clinician"})
    assert response.status_code == 200, "demo login is required to reach validation"
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


# ─── POST /patients ──────────────────────────────────────────────────────────


def test_create_patient_missing_required_fields_returns_422(client, auth_headers) -> None:
    response = client.post(PATIENTS_URL, json={}, headers=auth_headers)
    assert response.status_code == 422


def test_create_patient_missing_name_returns_422(client, auth_headers) -> None:
    response = client.post(PATIENTS_URL, json={"id": "PT-001"}, headers=auth_headers)
    assert response.status_code == 422


def test_create_patient_missing_id_returns_422(client, auth_headers) -> None:
    response = client.post(PATIENTS_URL, json={"name": "Test Patient"}, headers=auth_headers)
    assert response.status_code == 422


def test_create_patient_null_required_fields_returns_422(client, auth_headers) -> None:
    response = client.post(
        PATIENTS_URL, json={"id": None, "name": None}, headers=auth_headers
    )
    assert response.status_code == 422


def test_create_patient_wrong_field_type_returns_422(client, auth_headers) -> None:
    response = client.post(
        PATIENTS_URL, json={"id": 12345, "name": 678}, headers=auth_headers
    )
    assert response.status_code == 422


def test_create_patient_wrong_container_type_returns_422(client, auth_headers) -> None:
    response = client.post(
        PATIENTS_URL,
        json={"id": ["PT-001"], "name": {"first": "Test"}},
        headers=auth_headers,
    )
    assert response.status_code == 422


def test_create_patient_array_body_returns_422(client, auth_headers) -> None:
    """An array where the model expects an object."""
    response = client.post(
        PATIENTS_URL, json=[{"id": "PT-001", "name": "Test"}], headers=auth_headers
    )
    assert response.status_code == 422


def test_create_patient_malformed_json_returns_422(client, auth_headers) -> None:
    """Syntactically invalid JSON never reaches the handler."""
    response = client.post(
        PATIENTS_URL,
        content=b'{"id": "PT-001", "name":',
        headers={**auth_headers, "Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_create_patient_optional_field_wrong_type_returns_422(client, auth_headers) -> None:
    response = client.post(
        PATIENTS_URL,
        json={"id": "PT-002", "name": "Test Patient", "cancer_stage": ["II"]},
        headers=auth_headers,
    )
    assert response.status_code == 422


def test_create_patient_with_a_valid_body_is_not_rejected(client, auth_headers) -> None:
    """Control: the 422s above come from the payload, not a broken route."""
    response = client.post(
        PATIENTS_URL,
        json={"id": "PT-VALID-001", "name": "Valid Control Patient"},
        headers=auth_headers,
    )
    assert response.status_code != 422


# ─── POST /patients/{patient_id}/chat ────────────────────────────────────────


def test_chat_missing_required_field_returns_422(client, auth_headers) -> None:
    response = client.post(CHAT_URL, json={}, headers=auth_headers)
    assert response.status_code == 422


def test_chat_null_message_returns_422(client, auth_headers) -> None:
    response = client.post(CHAT_URL, json={"message": None}, headers=auth_headers)
    assert response.status_code == 422


def test_chat_wrong_field_type_returns_422(client, auth_headers) -> None:
    response = client.post(CHAT_URL, json={"message": 42}, headers=auth_headers)
    assert response.status_code == 422


def test_chat_wrong_container_type_returns_422(client, auth_headers) -> None:
    response = client.post(CHAT_URL, json={"message": ["hello"]}, headers=auth_headers)
    assert response.status_code == 422


def test_chat_misspelled_field_returns_422(client, auth_headers) -> None:
    """A typo'd key leaves the required field missing."""
    response = client.post(CHAT_URL, json={"mesage": "hello"}, headers=auth_headers)
    assert response.status_code == 422


def test_chat_array_body_returns_422(client, auth_headers) -> None:
    response = client.post(CHAT_URL, json=["hello"], headers=auth_headers)
    assert response.status_code == 422


def test_chat_malformed_json_returns_422(client, auth_headers) -> None:
    response = client.post(
        CHAT_URL,
        content=b'{"message": ',
        headers={**auth_headers, "Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_chat_with_a_valid_body_is_not_rejected(client, auth_headers) -> None:
    """Control: a well-formed message is not a validation error."""
    response = client.post(
        CHAT_URL, json={"message": "What does my latest record show?"}, headers=auth_headers
    )
    assert response.status_code != 422


# ─── the 422 response is usable ──────────────────────────────────────────────


def test_validation_error_names_the_offending_field(client, auth_headers) -> None:
    """A 422 that does not say what was wrong is a dead end for the caller."""
    response = client.post(PATIENTS_URL, json={"id": "PT-003"}, headers=auth_headers)
    assert response.status_code == 422

    detail = response.json()["detail"]
    assert isinstance(detail, list) and detail
    assert any("name" in str(item.get("loc", "")) for item in detail)


def test_validation_error_does_not_echo_a_stack_trace(client, auth_headers) -> None:
    """Validation failures are expected input, not server errors."""
    response = client.post(PATIENTS_URL, json={"id": 1, "name": 2}, headers=auth_headers)
    assert response.status_code == 422

    body = response.text.lower()
    assert "traceback" not in body
    assert "file \"" not in body
