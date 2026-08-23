"""Malformed request bodies are rejected at the named API boundaries.

Covers the three boundaries called out for input-validation evidence:

    POST /patients                       -> PatientCreate
    POST /patients/{patient_id}/chat     -> PatientChatRequest
    GET  /patient-report/{patient_id}    -> no request body (see below)

The assertions target *rejection*, not schema shape. A route can keep its
`response_model` and its typed parameter while quietly being switched to
`payload: dict`, and every happy-path test would still pass. The guards at the
bottom of this file exist so that change fails here instead.

A note on `/patient-report`: it is a **GET** route and therefore has no request
body, so there is no body-validation 422 contract to assert. Writing one would
mean inventing behaviour the API does not define. What is asserted instead is
what the route genuinely contracts — that it declares no request body and is
not silently a write boundary — plus its path-parameter handling.

`tests/test_request_validation_boundaries.py` already covers auth, clinician
review, and platform tenancy. This file deliberately does not repeat those; it
covers the patient-facing boundaries and shares the same rejection helper
semantics.
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

# 422 is the contract for a malformed body. 401/403 are equally acceptable
# outcomes on a protected route: they prove the request was refused *before*
# unvalidated content reached business logic, which is the property under test.
REJECTED = {401, 403, 422}


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _assert_rejected(response, boundary: str) -> None:
    assert response.status_code in REJECTED, (
        f"{boundary}: malformed body returned {response.status_code}; it must be "
        "rejected (422) or refused before the handler (401/403), never accepted."
    )
    assert response.status_code != 500, (
        f"{boundary}: a malformed body produced a 500, which means it reached "
        "application code before being validated."
    )


# ─── POST /patients ──────────────────────────────────────────────────────────

_PATIENT_MALFORMED = [
    pytest.param({}, id="both-required-fields-missing"),
    pytest.param({"id": "PT-VAL-1"}, id="name-missing"),
    pytest.param({"name": "Test Patient"}, id="id-missing"),
    pytest.param({"id": None, "name": None}, id="required-fields-null"),
    pytest.param({"id": 12345, "name": 678}, id="wrong-scalar-type"),
    pytest.param({"id": ["PT-1"], "name": {"first": "A"}}, id="wrong-container-type"),
    pytest.param({"id": "PT-1", "name": "A", "cancer_stage": ["II"]}, id="optional-wrong-type"),
]


@pytest.mark.parametrize("payload", _PATIENT_MALFORMED)
def test_patient_creation_rejects_malformed_body(client: TestClient, payload: dict) -> None:
    _assert_rejected(client.post("/patients", json=payload), "POST /patients")


def test_patient_creation_rejects_array_where_object_expected(client: TestClient) -> None:
    _assert_rejected(client.post("/patients", json=[{"id": "PT-1", "name": "A"}]), "POST /patients")


def test_patient_creation_rejects_non_json_body(client: TestClient) -> None:
    response = client.post(
        "/patients",
        content=b"id=PT-1&name=A",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    _assert_rejected(response, "POST /patients (form body)")


# ─── POST /patients/{patient_id}/chat ────────────────────────────────────────

_CHAT_PATH = "/patients/PT-VALIDATION/chat"

_CHAT_MALFORMED = [
    pytest.param({}, id="message-missing"),
    pytest.param({"message": None}, id="message-null"),
    pytest.param({"message": 42}, id="message-wrong-scalar-type"),
    pytest.param({"message": ["hello"]}, id="message-wrong-container-type"),
    pytest.param({"mesage": "typo"}, id="required-field-typo"),
]


@pytest.mark.parametrize("payload", _CHAT_MALFORMED)
def test_patient_chat_rejects_malformed_body(client: TestClient, payload: dict) -> None:
    _assert_rejected(client.post(_CHAT_PATH, json=payload), "POST /patients/{id}/chat")


def test_patient_chat_rejects_array_where_object_expected(client: TestClient) -> None:
    _assert_rejected(client.post(_CHAT_PATH, json=["hello"]), "POST /patients/{id}/chat")


def test_patient_chat_rejects_non_json_body(client: TestClient) -> None:
    response = client.post(
        _CHAT_PATH,
        content=b"not json at all",
        headers={"Content-Type": "text/plain"},
    )
    _assert_rejected(response, "POST /patients/{id}/chat (text body)")


# ─── GET /patient-report/{patient_id} ────────────────────────────────────────


def test_patient_report_declares_no_request_body() -> None:
    """`/patient-report` is a read boundary, so it has no body to validate.

    Asserted explicitly rather than skipped: if this route ever gains a request
    body, it becomes a write boundary and needs the malformed-body coverage the
    other two have.
    """
    spec = app.openapi()
    report = spec["paths"]["/patient-report/{patient_id}"]
    assert set(report) & {"get"}, "/patient-report must remain a read boundary"
    assert "post" not in report and "put" not in report and "patch" not in report
    assert "requestBody" not in report["get"]


def test_patient_report_rejects_unknown_patient_without_leaking(client: TestClient) -> None:
    """A read boundary must refuse cleanly, not 500 on an unmatched identifier."""
    response = client.get("/patient-report/PT-DOES-NOT-EXIST")
    assert response.status_code != 500
    assert response.status_code in REJECTED | {404}


# ─── the contract these tests protect ────────────────────────────────────────


def test_named_boundaries_stay_bound_to_typed_models() -> None:
    """Swapping a model for `dict` removes validation while every happy-path
    test keeps passing. This asserts the binding itself."""
    import typing

    from backend.api.routers import patient

    for func_name, param in (("create_patient", "payload"),):
        func = getattr(patient, func_name)
        annotation = typing.get_type_hints(func)[param]
        assert annotation is not dict, f"{func_name} accepts an unvalidated dict body"
        assert hasattr(annotation, "model_fields"), (
            f"{func_name}.{param} resolves to {annotation!r}, which FastAPI will not validate"
        )


def test_openapi_publishes_typed_request_schemas() -> None:
    """A published `$ref` requestBody is what makes validation externally visible."""
    spec = app.openapi()
    for path in ("/patients", "/patients/{patient_id}/chat"):
        body = spec["paths"][path]["post"].get("requestBody")
        assert body, f"{path} publishes no requestBody schema"
        assert "$ref" in str(body["content"]["application/json"]["schema"]), (
            f"{path} does not reference a named model"
        )


def test_request_models_declare_their_required_fields() -> None:
    """Required-field metadata is what turns a missing key into a 422."""
    schemas = app.openapi()["components"]["schemas"]
    assert set(schemas["PatientCreate"]["required"]) >= {"id", "name"}
    assert set(schemas["PatientChatRequest"]["required"]) >= {"message"}
