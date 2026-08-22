"""Malformed request bodies must be rejected at the API boundary.

The repository declares 117 Pydantic request models, but until now exactly one
test asserted that a malformed body is actually rejected. That gap matters:
`response_model=` and a typed `payload:` parameter are easy to keep while
accidentally removing the thing that enforces them — a route quietly switched
to `payload: dict` still type-checks, still passes every happy-path test, and
silently starts accepting anything.

These tests assert the *rejection*, not the schema. They cover the five request
boundaries where an unvalidated body would matter most:

    auth              credential and role selection
    patient           patient record creation
    clinician review  free-text clinical question
    platform          tenant/organization creation
    automation        job submission (already covered; kept adjacent)

Validation failures must surface as 422 from FastAPI's own handler rather than
as a 500 from the unhandled-exception path — a 500 would mean the body reached
application code before being checked.

Authorization is deliberately not exercised here: request validation runs
before the endpoint body, so a malformed payload is rejected whether or not the
caller is authorized. Where a route is protected, 401/403 is an acceptable
outcome and is asserted as such, because that too proves the request never
reached unvalidated business logic.
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

REJECTED = {401, 403, 422}


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _assert_not_accepted(response, boundary: str) -> None:
    """A malformed body must never be processed as if it were valid."""
    assert response.status_code in REJECTED, (
        f"{boundary}: malformed body returned {response.status_code}. "
        "It must be rejected (422), or refused before reaching the handler "
        "(401/403) — never accepted, and never a 500, which would mean the "
        "payload reached application code unvalidated."
    )
    assert response.status_code != 500


# ─── auth ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "payload",
    [
        {},                                  # required `role` absent
        {"role": None},                      # required field explicitly null
        {"role": 123},                       # wrong scalar type
        {"role": ["clinician"]},             # wrong container type
        {"patient_id": "PT-1"},              # only the optional field supplied
    ],
)
def test_demo_login_rejects_malformed_bodies(client: TestClient, payload: dict) -> None:
    response = client.post("/auth/demo-login", json=payload)
    assert response.status_code == 422, (
        f"unauthenticated login endpoint must validate: got {response.status_code}"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},                                  # both credentials absent
        {"username": "someone"},             # password missing
        {"password": "secret"},              # username missing
        {"username": None, "password": None},
        {"username": 1, "password": 2},
    ],
)
def test_credential_login_rejects_malformed_bodies(client: TestClient, payload: dict) -> None:
    response = client.post("/auth/demo-credential-login", json=payload)
    assert response.status_code == 422


def test_login_rejects_non_json_body(client: TestClient) -> None:
    """A body that is not JSON at all must not reach the handler."""
    response = client.post(
        "/auth/demo-credential-login",
        content=b"username=admin&password=admin",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 422


def test_login_rejects_json_array_where_object_expected(client: TestClient) -> None:
    response = client.post("/auth/demo-login", json=["clinician"])
    assert response.status_code == 422


# ─── patient creation ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "payload",
    [
        {},                                    # `id` and `name` both required
        {"id": "PT-VAL-1"},                    # name missing
        {"name": "Test Patient"},              # id missing
        {"id": None, "name": None},
        {"id": ["PT-1"], "name": {"first": "A"}},
    ],
)
def test_patient_creation_rejects_malformed_bodies(client: TestClient, payload: dict) -> None:
    _assert_not_accepted(client.post("/patients", json=payload), "POST /patients")


# ─── clinician review ────────────────────────────────────────────────────────


@pytest.mark.parametrize("payload", [{}, {"question": None}, {"question": 42}, {"quesiton": "typo"}])
def test_timeline_question_rejects_malformed_bodies(client: TestClient, payload: dict) -> None:
    _assert_not_accepted(
        client.post("/patients/PT-VALIDATION/timeline-question", json=payload),
        "POST /patients/{id}/timeline-question",
    )


# ─── platform / control plane ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "payload",
    [
        {},                                    # `name` required
        {"name": "ab"},                        # violates min_length=3
        {"name": "x" * 121},                   # violates max_length=120
        {"name": "valid org", "slug": "ab"},   # slug violates min_length=3
        {"name": None},
    ],
)
def test_organization_creation_enforces_field_constraints(
    client: TestClient, payload: dict
) -> None:
    """Constrained fields must enforce their bounds, not just their type."""
    _assert_not_accepted(client.post("/platform/organizations", json=payload), "POST /platform/organizations")


# ─── the contract these tests protect ────────────────────────────────────────


def test_validated_boundaries_declare_typed_request_models() -> None:
    """The routes above must keep binding a model, not a bare dict.

    Swapping `payload: SomeModel` for `payload: dict` keeps every happy-path
    test green while removing validation entirely. This asserts the binding
    itself, so that change fails here.
    """
    import typing

    from backend.api.routers import auth, clinician_review, platform

    cases = [
        (auth.demo_login, "payload"),
        (auth.demo_credential_login, "payload"),
        (clinician_review.answer_patient_timeline_question_endpoint, "payload"),
        (platform.add_organization, "payload"),
    ]
    for func, param in cases:
        # `get_type_hints`, not `inspect.signature`: some routers use
        # `from __future__ import annotations`, which leaves the annotation as
        # the *string* "TimelineQuestionRequest" rather than the class.
        annotation = typing.get_type_hints(func)[param]
        assert annotation is not dict, f"{func.__name__} accepts an unvalidated dict body"
        assert hasattr(annotation, "model_fields"), (
            f"{func.__name__}.{param} resolves to {annotation!r}, which is not a "
            "Pydantic model, so FastAPI will not validate it"
        )


def test_openapi_publishes_the_request_schemas() -> None:
    """A declared requestBody schema is what makes validation externally visible."""
    spec = app.openapi()
    for path in ("/auth/demo-login", "/auth/demo-credential-login", "/platform/organizations"):
        body = spec["paths"][path]["post"].get("requestBody")
        assert body, f"{path} publishes no requestBody schema"
        assert "$ref" in str(body["content"]["application/json"]["schema"])
