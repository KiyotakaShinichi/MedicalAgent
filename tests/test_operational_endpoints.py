"""Operational surface: liveness, readiness, correlation, and their contracts.

These lock in the behaviour an operator or an orchestrator depends on, and the
OpenAPI metadata that makes it discoverable without reading the source. The
endpoints themselves predate these tests; what was missing was a declared
contract, so a static reader — or an external evaluator — could not tell the
probes existed.

Nothing here asserts anything clinical. `/health` and `/ready` are deliberately
outside the medical surface.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.main import app  # noqa: E402
from backend.services.request_context import reset_request_id, set_request_id  # noqa: E402
from backend.services.runtime_health import application_version  # noqa: E402
from backend.services.structured_logging import (  # noqa: E402
    JsonEventFormatter,
    build_event,
    logging_config,
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# ─── Liveness ────────────────────────────────────────────────────────────────


def test_health_endpoint_exists_and_reports_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "nlcare_monitoring_prototype",
        "version": application_version(),
    }


def test_health_schema_is_stable(client: TestClient) -> None:
    """Exact key set. A probe consumer breaks on an unannounced field change.

    `version` was added deliberately, so an operator can tell *which build*
    answered a probe rather than only that one did. The assertion stays an
    exact-set comparison: the point is that the shape never drifts silently.
    """
    assert set(client.get("/health").json()) == {"status", "service", "version"}


def test_healthz_alias_matches_health(client: TestClient) -> None:
    """Orchestrators probing the Kubernetes-conventional path get the same answer."""
    assert client.get("/healthz").json() == client.get("/health").json()


def test_liveness_does_not_depend_on_the_database(client: TestClient) -> None:
    """Liveness must stay cheap: it decides restarts, not traffic routing.

    A liveness probe that consults the database turns a slow dependency into a
    restart loop. Asserting on the payload shape is the proxy — the response
    reports no dependency state at all.
    """
    payload = client.get("/health").json()
    assert "checks" not in payload
    assert "database" not in json.dumps(payload)


# ─── Readiness ───────────────────────────────────────────────────────────────


def test_ready_endpoint_reports_deterministic_contract(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code in (200, 503)
    payload = response.json()
    assert set(payload) == {
        "status",
        "service",
        "environment",
        "demo_auth_allowed",
        "checks",
        "clinical_validation",
        "healthcare_production_ready",
        "claim_boundary",
    }
    assert payload["status"] in {"ready", "not_ready"}
    # Readiness must never imply clinical clearance.
    assert payload["clinical_validation"] is False
    assert payload["healthcare_production_ready"] is False


def test_ready_status_code_tracks_readiness(client: TestClient) -> None:
    """503 when not ready, so a load balancer drains rather than restarts."""
    response = client.get("/ready")
    payload = response.json()
    expected = 200 if payload["status"] == "ready" else 503
    assert response.status_code == expected


def test_readyz_alias_matches_ready(client: TestClient) -> None:
    assert set(client.get("/readyz").json()) == set(client.get("/ready").json())


def test_readiness_probe_failure_is_handled_without_leaking_details() -> None:
    """A failing dependency must degrade, not raise, and must not leak the message.

    Exception *messages* routinely carry connection strings and filesystem
    paths; the class name is enough to triage.
    """
    from backend.services.runtime_health import readiness_payload

    class ExplodingSession:
        def execute(self, *_args, **_kwargs):
            raise RuntimeError("postgres://user:hunter2@db.internal:5432/nlcare")

    payload, ready = readiness_payload(
        ExplodingSession(),
        environment={"ENVIRONMENT": "test"},
        retrieval_probe=lambda: {"meets_deployment_requirement": True},
        demo_auth_probe=lambda: False,
    )
    assert ready is False
    assert payload["status"] == "not_ready"
    assert payload["checks"]["database"] == {"ready": False, "error_type": "RuntimeError"}
    serialized = json.dumps(payload)
    assert "hunter2" not in serialized
    assert "postgres://" not in serialized


# ─── OpenAPI detectability ───────────────────────────────────────────────────


def test_openapi_exposes_operational_probes_discoverably() -> None:
    """The regression this sprint exists to prevent.

    An external reader that only sees the OpenAPI document must be able to
    conclude the service has health and readiness probes. Previously both
    operations were untagged, carried an auto-derived summary, and declared an
    empty `schema: {}` response, so there was nothing to key on.
    """
    spec = app.openapi()
    for path in ("/health", "/ready"):
        operation = spec["paths"][path]["get"]
        assert "operations" in operation["tags"], f"{path} must be tagged"
        assert operation["summary"], f"{path} needs an explicit summary"
        assert operation["description"], f"{path} needs a description"
        schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
        assert schema, f"{path} must declare a response schema, not an empty one"
        assert "$ref" in json.dumps(schema)

    assert any(tag["name"] == "operations" for tag in spec.get("tags", []))
    assert spec["paths"]["/ready"]["get"]["responses"].get("503"), (
        "the not-ready status must be part of the published contract"
    )


def test_operational_probes_require_no_authentication() -> None:
    """A probe behind auth cannot be used by an orchestrator."""
    spec = app.openapi()
    for path in ("/health", "/ready"):
        assert not spec["paths"][path]["get"].get("security")


# ─── Structured logging ──────────────────────────────────────────────────────


def test_logging_config_is_declarative_and_uses_stdlib() -> None:
    config = logging_config()
    assert config["version"] == 1
    assert config["disable_existing_loggers"] is False
    assert config["handlers"]["nlcare_stdout"]["class"] == "logging.StreamHandler"
    assert "nlcare.events" in config["loggers"]


def test_structured_record_is_valid_json_with_expected_fields() -> None:
    record = logging.LogRecord(
        name="nlcare.events", level=logging.INFO, pathname=__file__,
        lineno=1, msg="patient_report_generated", args=(), exc_info=None,
    )
    record.nlcare_event = build_event("patient_report_generated", component="api")
    emitted = json.loads(JsonEventFormatter().format(record))
    for field in ("schema_version", "event_type", "severity", "component", "timestamp"):
        assert field in emitted


def test_event_inherits_the_in_flight_request_id() -> None:
    """Service-layer events must correlate with the request that caused them.

    Before this, a `log_event` call outside the middleware minted a fresh id,
    so events from one request could not be joined together.
    """
    token = set_request_id("req_operational_test")
    try:
        event = build_event("rag_query_served", component="rag")
    finally:
        reset_request_id(token)
    assert event["request_id"] == "req_operational_test"
    assert event["correlation_id"] == "req_operational_test"


def test_event_still_mints_an_id_outside_a_request() -> None:
    """Background work has no ambient request; it must still be traceable."""
    event = build_event("scheduled_job_finished", component="worker")
    assert event["request_id"]


def test_explicit_request_id_wins_over_ambient() -> None:
    token = set_request_id("ambient")
    try:
        event = build_event("thing", request_id="explicit")
    finally:
        reset_request_id(token)
    assert event["request_id"] == "explicit"


def test_sensitive_fields_are_never_emitted() -> None:
    """Redaction covers the fields most likely to carry regulated content.

    `api_key` and `apiKey` are here because a redaction probe caught them
    reaching the log intact: the key-part list had `secret` and `token` but
    nothing matching `api_key`.
    """
    event = build_event(
        "patient_chat_turn",
        component="api",
        patient_id="PT-000123",
        details={
            "authorization": "Bearer sk-live-abcdef123456",
            "api_token": "secret-value",
            "api_key": "k-live-should-not-appear",
            "apiKey": "camel-case-variant",
            "credential": "cred-should-not-appear",
            "message": "I found a lump in my left breast last Tuesday",
            "prompt": "patient asked about metastasis",
            "nested": {"password": "hunter2", "patient": "Maria Santos"},
            "safe_field": "kept",
        },
    )
    serialized = json.dumps(event)
    for secret in (
        "Bearer sk-live-abcdef123456",
        "secret-value",
        "k-live-should-not-appear",
        "camel-case-variant",
        "cred-should-not-appear",
        "lump in my left breast",
        "hunter2",
        "Maria Santos",
        "PT-000123",
    ):
        assert secret not in serialized, f"{secret!r} leaked into the structured log"
    assert event["details"]["safe_field"] == "kept"
    assert event["patient_id"] == "[REDACTED]"


def test_identifiers_inside_innocently_named_fields_are_redacted() -> None:
    """Key-name matching cannot catch an identifier in a benign field.

    Value-pattern redaction is the second layer, shared with the database
    audit trail so both logging paths enforce one policy.
    """
    event = build_event(
        "report_generated",
        component="api",
        details={"summary": "contact maria@example.com or 555-123-4567"},
    )
    summary = event["details"]["summary"]
    assert "maria@example.com" not in summary
    assert "555-123-4567" not in summary
    assert "[redacted]" in summary


def test_operational_fields_survive_redaction() -> None:
    """Over-redaction is its own failure: these are what you debug with.

    A bare `key` substring would blank all three of the `*_key` fields below.
    """
    event = build_event(
        "http_request_completed",
        component="api",
        details={
            "route": "/patients/{patient_id}/labs",
            "method": "GET",
            "status_code": 200,
            "duration_ms": 12.5,
            "cache_key": "kb-9f2",
            "idempotency_key": "idem-1",
        },
    )
    details = event["details"]
    assert details["route"] == "/patients/{patient_id}/labs"
    assert details["method"] == "GET"
    assert details["status_code"] == 200
    assert details["duration_ms"] == 12.5
    assert details["cache_key"] == "kb-9f2"
    assert details["idempotency_key"] == "idem-1"


# ─── Request correlation ─────────────────────────────────────────────────────


def test_request_id_is_returned_and_echoed(client: TestClient) -> None:
    supplied = "req-correlation-check-001"
    response = client.get("/health", headers={"X-Request-ID": supplied})
    assert response.headers["x-request-id"] == supplied


def test_request_id_is_generated_when_absent(client: TestClient) -> None:
    response = client.get("/health")
    assert response.headers.get("x-request-id")


def test_request_id_is_exposed_to_browsers() -> None:
    """CORS must expose the header or a browser client cannot read it back."""
    from backend.api.main import app as application

    exposed = [
        m for m in application.user_middleware if "CORSMiddleware" in str(m)
    ]
    assert exposed, "CORS middleware must be installed"
    assert "X-Request-ID" in str(exposed[0])


def test_unhandled_exception_returns_correlated_non_revealing_error() -> None:
    """An unhandled error must be traceable without disclosing its contents."""
    from backend.api.main import app as application

    secret = "postgres://user:hunter2@db.internal/nlcare"

    @application.get("/__operational_boom", include_in_schema=False)
    def _boom():
        raise RuntimeError(secret)

    try:
        probe = TestClient(application, raise_server_exceptions=False)
        response = probe.get("/__operational_boom", headers={"X-Request-ID": "req-boom-1"})
        assert response.status_code == 500
        payload = response.json()
        assert payload["error"] == "internal_server_error"
        assert payload["request_id"] == "req-boom-1"
        assert response.headers["x-request-id"] == "req-boom-1"

        body = json.dumps(payload)
        assert secret not in body
        assert "hunter2" not in body
        assert "RuntimeError" not in body
        assert "Traceback" not in body
    finally:
        application.router.routes = [
            route
            for route in application.router.routes
            if getattr(route, "path", None) != "/__operational_boom"
        ]
