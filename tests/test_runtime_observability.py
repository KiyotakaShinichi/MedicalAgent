"""Runtime observability must be correlated, redacted, bounded, and optional."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fastapi import Request
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.services.error_reporting import (
    OperationalErrorCategory,
    capture_exception,
    set_error_reporter,
)
from backend.services.request_context import (
    MAX_REQUEST_ID_LENGTH,
    get_request_id,
    normalize_request_id,
)
from backend.services.pii_redaction import redact_payload
from backend.services.runtime_metrics import (
    InMemoryRuntimeMetrics,
    record_http_request,
    set_runtime_metrics_sink,
)
from backend.services.structured_logging import LOGGER, build_event, log_event


SENTINEL = "R4_SENTINEL_MUST_NOT_APPEAR_7f8a9b"


class _CollectingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        event = getattr(record, "nlcare_event", None)
        if isinstance(event, dict):
            self.events.append(event)


@contextmanager
def _temporary_route(path: str, endpoint) -> Iterator[None]:
    app.get(path, include_in_schema=False)(endpoint)
    try:
        yield
    finally:
        app.router.routes = [
            route for route in app.router.routes if getattr(route, "path", None) != path
        ]


def test_sensitive_sentinels_never_reach_structured_event_json() -> None:
    event = build_event(
        "redaction_probe",
        details={
            "authorization": f"Bearer {SENTINEL}",
            "cookie": f"session={SENTINEL}",
            "session_token": SENTINEL,
            "api_key": SENTINEL,
            "webhook_secret": SENTINEL,
            "patient_payload": {"message": SENTINEL},
            "safe_url": f"/callback?token={SENTINEL}",
        },
    )

    serialized = json.dumps(event)
    assert SENTINEL not in serialized
    assert event["service"] == "nlcare_monitoring_prototype"
    assert event["event"] == event["event_type"] == "redaction_probe"
    assert event["level"] == event["severity"] == "info"


def test_secret_patterns_are_redacted_even_under_an_innocent_key() -> None:
    event = build_event(
        "redaction_probe",
        details={"summary": "Authorization failed for Bearer abcdefghijklmnop123456"},
    )
    assert "abcdefghijklmnop123456" not in json.dumps(event)


def test_database_audit_payload_redaction_drops_patient_prose_and_secrets() -> None:
    redacted = redact_payload(
        {
            "symptom": SENTINEL,
            "notes": SENTINEL,
            "findings": SENTINEL,
            "authorization": f"Bearer {SENTINEL}",
            "warning_count": 2,
        }
    )
    assert SENTINEL not in json.dumps(redacted)
    assert redacted["warning_count"] == 2


def test_request_id_normalization_preserves_only_bounded_safe_values() -> None:
    assert normalize_request_id("gateway-req_123:child") == "gateway-req_123:child"

    for invalid in ("line\nbreak", "contains spaces", "x" * 129, "", None):
        normalized = normalize_request_id(invalid)
        assert normalized.startswith("req_")
        assert len(normalized) <= MAX_REQUEST_ID_LENGTH
        assert normalized != invalid

    event = build_event("bounded", request_id=SENTINEL + " " + ("x" * 200))
    assert event["request_id"].startswith("req_")
    assert SENTINEL not in json.dumps(event)


def test_middleware_router_log_and_response_share_one_request_id() -> None:
    handler = _CollectingHandler()
    previous_level = LOGGER.level
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(handler)

    def correlated(request: Request) -> dict[str, str | None]:
        log_event("router_correlation_probe", component="test")
        return {
            "context_request_id": get_request_id(),
            "state_request_id": request.state.request_id,
        }

    try:
        with _temporary_route("/__r4_correlation", correlated):
            response = TestClient(app).get(
                "/__r4_correlation",
                headers={"X-Request-ID": "r4-correlation-001"},
            )
    finally:
        LOGGER.removeHandler(handler)
        LOGGER.setLevel(previous_level)

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "r4-correlation-001"
    assert response.json() == {
        "context_request_id": "r4-correlation-001",
        "state_request_id": "r4-correlation-001",
    }
    router_event = next(e for e in handler.events if e["event"] == "router_correlation_probe")
    assert router_event["request_id"] == "r4-correlation-001"


def test_invalid_inbound_request_id_is_not_reflected() -> None:
    supplied = SENTINEL + "\r\nX-Forged: yes"
    response = TestClient(app).get("/health", headers={"X-Request-ID": supplied})
    returned = response.headers["x-request-id"]
    assert returned.startswith("req_")
    assert SENTINEL not in returned
    assert len(returned) <= MAX_REQUEST_ID_LENGTH


class _CollectingReporter:
    def __init__(self) -> None:
        self.exceptions: list[dict[str, Any]] = []

    def capture_exception(self, error, *, category, request_id=None, context=None) -> None:
        self.exceptions.append(
            {
                "error_type": type(error).__name__,
                "category": category,
                "request_id": request_id,
                "context": context,
            }
        )

    def capture_message(self, message, *, category, request_id=None, context=None) -> None:
        return None


class _ExplodingReporter(_CollectingReporter):
    def capture_exception(self, error, *, category, request_id=None, context=None) -> None:
        raise RuntimeError("reporter unavailable")


def test_unhandled_error_is_classified_reported_and_bounded() -> None:
    reporter = _CollectingReporter()
    previous = set_error_reporter(reporter)

    def boom() -> None:
        raise RuntimeError(f"database password={SENTINEL}")

    try:
        with _temporary_route("/__r4_error", boom):
            response = TestClient(app, raise_server_exceptions=False).get(
                "/__r4_error",
                headers={"X-Request-ID": "r4-error-001"},
            )
    finally:
        set_error_reporter(previous)

    assert response.status_code == 500
    assert response.json()["error"] == "internal_server_error"
    assert response.json()["request_id"] == "r4-error-001"
    assert response.headers["x-request-id"] == "r4-error-001"
    assert SENTINEL not in response.text
    assert "Traceback" not in response.text
    assert reporter.exceptions == [
        {
            "error_type": "RuntimeError",
            "category": OperationalErrorCategory.INTERNAL_SERVICE,
            "request_id": "r4-error-001",
            "context": {"method": "GET", "route": "/__r4_error"},
        }
    ]


def test_error_reporter_failure_never_changes_safe_response() -> None:
    previous = set_error_reporter(_ExplodingReporter())

    def boom() -> None:
        raise RuntimeError(SENTINEL)

    try:
        with _temporary_route("/__r4_reporter_failure", boom):
            response = TestClient(app, raise_server_exceptions=False).get(
                "/__r4_reporter_failure"
            )
    finally:
        set_error_reporter(previous)

    assert response.status_code == 500
    assert response.json()["error"] == "internal_server_error"
    assert SENTINEL not in response.text


class _ExplodingMetrics:
    def record_http(self, metric) -> None:
        raise RuntimeError("metrics unavailable")

    def record_readiness(self, *, ready: bool) -> None:
        raise RuntimeError("metrics unavailable")


def test_metrics_failure_never_changes_application_behavior() -> None:
    previous = set_runtime_metrics_sink(_ExplodingMetrics())
    try:
        response = TestClient(app).get("/health")
        record_http_request(method="GET", route="/probe", status_code=200, duration_ms=1.0)
    finally:
        set_runtime_metrics_sink(previous)
    assert response.status_code == 200


def test_runtime_metrics_are_aggregated_without_high_cardinality_labels() -> None:
    sink = InMemoryRuntimeMetrics()
    previous = set_runtime_metrics_sink(sink)
    try:
        record_http_request(
            method="get",
            route="/patients/{patient_id}/labs",
            status_code=200,
            duration_ms=12.5,
        )
        record_http_request(
            method="get",
            route="/patients/{patient_id}/labs",
            status_code=503,
            duration_ms=25.0,
        )
    finally:
        set_runtime_metrics_sink(previous)

    snapshot = sink.snapshot()
    assert snapshot["http_error_count"] == 1
    assert snapshot["latency"] == {"count": 2, "mean_ms": 18.75, "max_ms": 25.0}
    serialized = json.dumps(snapshot)
    assert "request_id" not in serialized
    assert "P001" not in serialized


def test_capture_exception_guard_can_be_called_without_a_provider() -> None:
    previous = set_error_reporter(_ExplodingReporter())
    try:
        capture_exception(RuntimeError(SENTINEL))
    finally:
        set_error_reporter(previous)
