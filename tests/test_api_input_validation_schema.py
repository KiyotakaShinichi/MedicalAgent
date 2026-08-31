"""Schema-level validation on the patient write boundaries.

`backend/api/schemas/patient.py` declares the structural bounds - length,
numeric range, type - as Pydantic field constraints, so a violating request is
rejected before any handler code runs.

Status codes, and why
---------------------
Every schema violation here returns **422**, which is what FastAPI returns when
a request body does not satisfy its model. The repository also has a **400**
contract, returned by `backend.services.input_validation` with a structured
`validation_error_payload`; that layer is still present and still enforces the
same bounds for callers that never send a request body, notably the support
agent's record-write actions.

Moving the bounds into the schema means an HTTP request that breaches one is
now refused at the boundary with 422 rather than reaching the handler and being
refused with 400. Accepted inputs are unchanged - the valid-control tests below
pin that. The bounds themselves are imported from `input_validation` rather
than duplicated, so the two layers cannot drift into different answers.

`tests/test_api_input_validation.py` covers the clinician-facing `/patients`
boundaries; this file covers the patient-facing `/me` routes DataFactor names.
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
from backend.services.input_validation import (  # noqa: E402
    CBC_LIMITS,
    CHAT_MESSAGE_MAX_LENGTH,
    SEVERITY_MAX,
    SYMPTOM_MAX_LENGTH,
)

SYMPTOMS_URL = "/me/symptoms"
LABS_URL = "/me/labs"
CHAT_URL = "/me/chat"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def auth_headers(client: TestClient) -> dict[str, str]:
    """A demo patient token.

    Authorization is checked before the body is validated, so without this
    every request below would return 401 and prove nothing about validation.
    """
    response = client.post("/auth/demo-login", json={"role": "patient", "patient_id": "P001"})
    assert response.status_code == 200, "demo login is required to reach validation"
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _valid_symptom() -> dict:
    return {"date": "2026-01-01", "symptom": "fatigue", "severity": 3}


def _valid_labs() -> dict:
    return {"date": "2026-01-01", "wbc": 6.0, "hemoglobin": 12.0, "platelets": 200.0}


# ─── POST /me/symptoms ───────────────────────────────────────────────────────


def test_symptoms_missing_required_fields_returns_422(client, auth_headers) -> None:
    response = client.post(SYMPTOMS_URL, json={}, headers=auth_headers)
    assert response.status_code == 422


def test_symptoms_missing_severity_returns_422(client, auth_headers) -> None:
    response = client.post(
        SYMPTOMS_URL, json={"date": "2026-01-01", "symptom": "fatigue"}, headers=auth_headers
    )
    assert response.status_code == 422


def test_symptoms_wrong_severity_type_returns_422(client, auth_headers) -> None:
    payload = {**_valid_symptom(), "severity": "very bad"}
    response = client.post(SYMPTOMS_URL, json=payload, headers=auth_headers)
    assert response.status_code == 422


def test_symptoms_severity_above_range_returns_422(client, auth_headers) -> None:
    payload = {**_valid_symptom(), "severity": SEVERITY_MAX + 1}
    response = client.post(SYMPTOMS_URL, json=payload, headers=auth_headers)
    assert response.status_code == 422


def test_symptoms_negative_severity_returns_422(client, auth_headers) -> None:
    payload = {**_valid_symptom(), "severity": -1}
    response = client.post(SYMPTOMS_URL, json=payload, headers=auth_headers)
    assert response.status_code == 422


def test_symptoms_oversize_label_returns_422(client, auth_headers) -> None:
    payload = {**_valid_symptom(), "symptom": "x" * (SYMPTOM_MAX_LENGTH + 1)}
    response = client.post(SYMPTOMS_URL, json=payload, headers=auth_headers)
    assert response.status_code == 422


def test_symptoms_blank_label_returns_422(client, auth_headers) -> None:
    """Whitespace satisfies a length check but records nothing."""
    payload = {**_valid_symptom(), "symptom": "   "}
    response = client.post(SYMPTOMS_URL, json=payload, headers=auth_headers)
    assert response.status_code == 422


def test_symptoms_oversize_notes_returns_422(client, auth_headers) -> None:
    payload = {**_valid_symptom(), "notes": "n" * 5000}
    response = client.post(SYMPTOMS_URL, json=payload, headers=auth_headers)
    assert response.status_code == 422


def test_symptoms_invalid_date_returns_422(client, auth_headers) -> None:
    payload = {**_valid_symptom(), "date": "not-a-date"}
    response = client.post(SYMPTOMS_URL, json=payload, headers=auth_headers)
    assert response.status_code == 422


def test_symptoms_malformed_body_returns_422(client, auth_headers) -> None:
    response = client.post(
        SYMPTOMS_URL,
        content=b'{"date": "2026-01-01", "symptom":',
        headers={**auth_headers, "Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_symptoms_valid_payload_is_accepted(client, auth_headers) -> None:
    """Control: the rejections above come from the payload, not a broken route."""
    response = client.post(SYMPTOMS_URL, json=_valid_symptom(), headers=auth_headers)
    assert response.status_code == 200


def test_symptoms_boundary_severity_values_are_accepted(client, auth_headers) -> None:
    """The range is inclusive; the edges must not be rejected."""
    for severity in (0, SEVERITY_MAX):
        payload = {**_valid_symptom(), "severity": severity}
        response = client.post(SYMPTOMS_URL, json=payload, headers=auth_headers)
        assert response.status_code == 200, f"severity {severity} should be accepted"


# ─── POST /me/labs ───────────────────────────────────────────────────────────


def test_labs_missing_required_fields_returns_422(client, auth_headers) -> None:
    response = client.post(LABS_URL, json={}, headers=auth_headers)
    assert response.status_code == 422


def test_labs_wrong_type_returns_422(client, auth_headers) -> None:
    payload = {**_valid_labs(), "wbc": "high"}
    response = client.post(LABS_URL, json=payload, headers=auth_headers)
    assert response.status_code == 422


def test_labs_negative_wbc_returns_422(client, auth_headers) -> None:
    payload = {**_valid_labs(), "wbc": -5.0}
    response = client.post(LABS_URL, json=payload, headers=auth_headers)
    assert response.status_code == 422


@pytest.mark.parametrize("field", ["wbc", "hemoglobin", "platelets"])
def test_labs_value_above_accepted_maximum_returns_422(client, auth_headers, field) -> None:
    payload = {**_valid_labs(), field: CBC_LIMITS[field]["max"] * 10}
    response = client.post(LABS_URL, json=payload, headers=auth_headers)
    assert response.status_code == 422


@pytest.mark.parametrize("field", ["wbc", "hemoglobin", "platelets"])
def test_labs_value_below_accepted_minimum_returns_422(client, auth_headers, field) -> None:
    payload = {**_valid_labs(), field: 0.0}
    response = client.post(LABS_URL, json=payload, headers=auth_headers)
    assert response.status_code == 422


def test_labs_wrong_structure_returns_422(client, auth_headers) -> None:
    """An array where the model expects an object."""
    response = client.post(LABS_URL, json=[_valid_labs()], headers=auth_headers)
    assert response.status_code == 422


def test_labs_malformed_body_returns_422(client, auth_headers) -> None:
    response = client.post(
        LABS_URL,
        content=b'{"wbc": 6.0,',
        headers={**auth_headers, "Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_labs_valid_payload_is_accepted(client, auth_headers) -> None:
    response = client.post(LABS_URL, json=_valid_labs(), headers=auth_headers)
    assert response.status_code == 200


def test_labs_clinically_alarming_but_in_range_values_are_accepted(client, auth_headers) -> None:
    """Accepted-with-warning is a real state and must not become a rejection.

    A very low white count is inside the demo bounds and is exactly the case
    the warning layer exists for. Rejecting it would discard the record the
    care team most needs to see.
    """
    payload = {**_valid_labs(), "wbc": CBC_LIMITS["wbc"]["watch_low"] - 0.5}
    response = client.post(LABS_URL, json=payload, headers=auth_headers)
    assert response.status_code == 200


# ─── POST /me/chat ───────────────────────────────────────────────────────────


def test_chat_missing_message_returns_422(client, auth_headers) -> None:
    response = client.post(CHAT_URL, json={}, headers=auth_headers)
    assert response.status_code == 422


def test_chat_empty_message_returns_422(client, auth_headers) -> None:
    response = client.post(CHAT_URL, json={"message": ""}, headers=auth_headers)
    assert response.status_code == 422


def test_chat_blank_message_returns_422(client, auth_headers) -> None:
    response = client.post(CHAT_URL, json={"message": "    "}, headers=auth_headers)
    assert response.status_code == 422


def test_chat_oversize_message_returns_422(client, auth_headers) -> None:
    payload = {"message": "x" * (CHAT_MESSAGE_MAX_LENGTH + 1)}
    response = client.post(CHAT_URL, json=payload, headers=auth_headers)
    assert response.status_code == 422


def test_chat_wrong_message_type_returns_422(client, auth_headers) -> None:
    response = client.post(CHAT_URL, json={"message": 42}, headers=auth_headers)
    assert response.status_code == 422


def test_chat_malformed_body_returns_422(client, auth_headers) -> None:
    response = client.post(
        CHAT_URL,
        content=b'{"message":',
        headers={**auth_headers, "Content-Type": "application/json"},
    )
    assert response.status_code == 422


# ─── the schema is the single declaration of these bounds ────────────────────


def test_schema_bounds_come_from_the_shared_declaration() -> None:
    """Two copies of the numbers would eventually disagree.

    The schema imports the bounds from `input_validation`, which is also the
    layer the support agent's record-write path uses. This asserts the import
    rather than the values, because the point is that there is one source.
    """
    source = (ROOT / "backend/api/schemas/patient.py").read_text(encoding="utf-8")
    assert "from backend.services.input_validation import" in source
    for name in ("SYMPTOM_MAX_LENGTH", "SEVERITY_MAX", "CHAT_MESSAGE_MAX_LENGTH", "CBC_LIMITS"):
        assert name in source, f"{name} is not sourced from the shared declaration"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_cbc_schema_rejects_non_finite_numbers(value: float) -> None:
    from pydantic import ValidationError

    from backend.api.schemas.patient import MyLabCreate

    with pytest.raises(ValidationError):
        MyLabCreate(
            date="2026-01-01",
            wbc=value,
            hemoglobin=12.0,
            platelets=200.0,
        )


@pytest.mark.parametrize(
    ("model_name", "payload"),
    [
        (
            "MySymptomCreate",
            {"date": "2026-01-01", "symptom": "fatigue", "severity": 3, "typo": True},
        ),
        (
            "MyLabCreate",
            {
                "date": "2026-01-01",
                "wbc": 6.0,
                "hemoglobin": 12.0,
                "platelets": 200.0,
                "unknown_lab": 1,
            },
        ),
        ("PatientChatRequest", {"message": "What does WBC mean?", "instruction": "hidden"}),
    ],
)
def test_patient_write_schemas_reject_unknown_fields(model_name: str, payload: dict) -> None:
    from pydantic import ValidationError

    from backend.api.schemas import patient

    with pytest.raises(ValidationError):
        getattr(patient, model_name).model_validate(payload)


def test_openapi_publishes_the_constraints() -> None:
    """Declared constraints must be visible to a client reading the schema."""
    schemas = app.openapi()["components"]["schemas"]

    symptom = schemas["MySymptomCreate"]["properties"]
    assert symptom["severity"]["maximum"] == SEVERITY_MAX
    assert symptom["symptom"]["maxLength"] == SYMPTOM_MAX_LENGTH

    chat = schemas["PatientChatRequest"]["properties"]
    assert chat["message"]["maxLength"] == CHAT_MESSAGE_MAX_LENGTH

    labs = schemas["MyLabCreate"]["properties"]
    assert labs["wbc"]["maximum"] == CBC_LIMITS["wbc"]["max"]


def test_domain_validators_still_enforce_the_same_bounds() -> None:
    """The non-HTTP caller is still protected.

    The support agent builds symptom and lab records from a conversation and
    never passes through a request body, so the schema cannot protect it. This
    is why both layers exist.
    """
    from backend.services.input_validation import validate_cbc_values, validate_symptom_payload

    with pytest.raises(ValueError):
        validate_symptom_payload("fatigue", SEVERITY_MAX + 1)
    with pytest.raises(ValueError):
        validate_cbc_values(-5.0, 12.0, 200.0)

    # And an in-range value still returns warnings rather than raising.
    warnings = validate_cbc_values(CBC_LIMITS["wbc"]["watch_low"] - 0.5, 12.0, 200.0)
    assert any(w["level"] == "clinician_review" for w in warnings)
