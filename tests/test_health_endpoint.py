"""Operational contract: `GET /health`, `/ready`, and the logging pipeline.

The liveness/readiness split matters and is deliberate, so it is asserted
rather than assumed:

* **`/health`** answers "is this process alive?" It reports `status`,
  `service`, `version`, and database reachability. The database result is
  **informational**: the endpoint returns 200 with `status: ok` even when the
  database is unreachable. An orchestrator uses liveness to decide whether to
  *restart* the process, and restarting cannot repair a database — a probe that
  failed on a dependency outage would turn that outage into a restart loop. The
  probe is also bounded, because a liveness probe that hangs is a restart
  vector too.
* **`/ready`** answers "should traffic be sent here?" It aggregates database,
  retrieval index, and (when shared rate limiting is on) Redis, and returns 503
  when any required one is unavailable, so a load balancer drains the instance
  instead of restarting it. It stays the authoritative, fail-closed signal.

The second half of this file covers structured logging. The two belong
together: the probe says whether the process is alive, the logs say what it
did, and both have to be conventional enough for a tool — not only a person —
to find.

`tests/test_operational_endpoints.py` covers correlation, OpenAPI
discoverability, and log redaction at the middleware level. This file covers
the probe response contract and the logging configuration itself.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.deps import get_db  # noqa: E402
from backend.api.main import app  # noqa: E402
from backend.services.runtime_health import (  # noqa: E402
    application_version,
    database_connectivity,
    liveness_payload,
    readiness_payload,
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@contextmanager
def _unreachable_database():
    """A client whose database session fails the way a dead database does.

    The error text deliberately carries a credential-bearing DSN, so the
    redaction assertions are testing against the shape of a real failure rather
    than a sanitised stand-in.
    """

    class UnreachableSession:
        def execute(self, *_args, **_kwargs):
            raise OperationalError(
                "SELECT 1",
                {},
                Exception("could not connect to postgres://user:hunter2@db.internal:5432/nlcare_prod"),
            )

        def close(self) -> None:
            return None

    app.dependency_overrides[get_db] = lambda: UnreachableSession()
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db, None)


# ─── GET /health ─────────────────────────────────────────────────────────────


def test_health_returns_200(client: TestClient) -> None:
    assert client.get("/health").status_code == 200


def test_health_exposes_the_expected_keys(client: TestClient) -> None:
    """Exact key set: a probe consumer breaks on an unannounced field change."""
    assert set(client.get("/health").json()) == {
        "status",
        "service",
        "version",
        "database",
        "rag_index",
    }


def test_health_reports_ok_status_and_service(client: TestClient) -> None:
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["service"] == "nlcare_monitoring_prototype"


def test_health_reports_a_version(client: TestClient) -> None:
    """Tells an operator *which build* answered, not merely that one did."""
    version = client.get("/health").json()["version"]
    assert isinstance(version, str) and version, "version must be a non-empty string"
    assert version == application_version()


def test_version_resolution_never_raises() -> None:
    """A liveness probe must not fail because version lookup did."""
    assert isinstance(application_version(), str)


# ─── /health database connectivity (informational) ───────────────────────────


def test_health_reports_database_connectivity(client: TestClient) -> None:
    """The field exists and is a boolean, so a reader can act on it."""
    database = client.get("/health").json()["database"]
    assert "connected" in database
    assert isinstance(database["connected"], bool)


def test_health_reports_connected_when_the_database_is_reachable(
    client: TestClient,
) -> None:
    assert client.get("/health").json()["database"]["connected"] is True


def test_health_stays_200_and_ok_when_the_database_is_unreachable() -> None:
    """The whole point of the liveness/readiness split, asserted directly.

    A live process with a dead database is still live. Failing this probe would
    make an orchestrator restart every replica, which cannot repair a database
    and takes the service down with it. The database result is informational:
    it changes the `database` field and nothing else.
    """
    with _unreachable_database() as failing_client:
        response = failing_client.get("/health")

    assert response.status_code == 200, "a dead database must not fail liveness"
    payload = response.json()
    assert payload["status"] == "ok", "liveness status must not track the database"
    assert payload["database"]["connected"] is False
    assert payload["service"] == "nlcare_monitoring_prototype"
    assert payload["version"] == application_version()


def test_health_reports_only_an_exception_class_for_a_failed_probe() -> None:
    """Never the message: it carries the DSN, and this route is unauthenticated."""
    with _unreachable_database() as failing_client:
        payload = failing_client.get("/health").json()

    assert payload["database"]["error_type"] == "OperationalError"
    serialized = json.dumps(payload)
    for secret in ("hunter2", "postgres://", "db.internal", "5432", "nlcare_prod"):
        assert secret not in serialized, f"/health leaked {secret!r}"


def test_health_database_probe_is_bounded() -> None:
    """A liveness probe that hangs is a restart vector as surely as one that 500s.

    A driver call that never returns would hold the response open past any
    orchestrator's probe timeout, so the probe carries its own deadline.
    """

    class HangingSession:
        def execute(self, *_args, **_kwargs):
            time.sleep(30)

    started = time.perf_counter()
    result = database_connectivity(HangingSession(), timeout_seconds=0.2)
    elapsed = time.perf_counter() - started

    assert elapsed < 5, f"probe took {elapsed:.1f}s; it must not block on a hung driver"
    assert result == {"connected": False, "error_type": "TimeoutError"}


def test_database_connectivity_never_raises() -> None:
    """Any driver failure must become a reported value, not an exception."""

    class ExplodingSession:
        def execute(self, *_args, **_kwargs):
            raise RuntimeError("postgres://user:hunter2@db.internal:5432/nlcare_prod")

    result = database_connectivity(ExplodingSession())
    assert result["connected"] is False
    assert result["error_type"] == "RuntimeError"
    assert "hunter2" not in json.dumps(result)


def test_health_does_not_report_full_readiness(client: TestClient) -> None:
    """`/health` reports one informational fact, not the readiness verdict.

    Readiness aggregates retrieval and Redis as well, and it is the endpoint
    allowed to say "do not send traffic here". Duplicating that decision onto
    liveness is what recreates the restart loop.
    """
    payload = client.get("/health").json()
    assert "checks" not in payload
    assert "retrieval" not in payload and "redis" not in payload


def test_healthz_alias_matches_health(client: TestClient) -> None:
    """Kubernetes-conventional path returns the same answer."""
    assert client.get("/healthz").json() == client.get("/health").json()


def test_health_requires_no_authentication() -> None:
    """A probe behind auth cannot be used by an orchestrator."""
    assert not app.openapi()["paths"]["/health"]["get"].get("security")


def test_health_publishes_a_response_schema() -> None:
    operation = app.openapi()["paths"]["/health"]["get"]
    schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert "$ref" in str(schema), "/health must declare a typed response model"
    assert "operations" in operation["tags"]


# ─── GET /ready — database connectivity ──────────────────────────────────────


def test_ready_reports_database_connectivity(client: TestClient) -> None:
    payload = client.get("/ready").json()
    assert "database" in payload["checks"], "readiness must report DB connectivity"
    assert isinstance(payload["checks"]["database"]["ready"], bool)


def test_ready_status_code_tracks_readiness(client: TestClient) -> None:
    """503 when a required dependency is down, so a balancer drains the node."""
    response = client.get("/ready")
    expected = 200 if response.json()["status"] == "ready" else 503
    assert response.status_code == expected


def test_ready_reports_not_ready_when_the_database_is_unreachable() -> None:
    """The unhealthy-DB contract, exercised directly against the probe."""

    class UnreachableSession:
        def execute(self, *_args, **_kwargs):
            raise RuntimeError("postgres://user:hunter2@db.internal:5432/nlcare")

    payload, ready = readiness_payload(
        UnreachableSession(),
        environment={"ENVIRONMENT": "test"},
        retrieval_probe=lambda: {"meets_deployment_requirement": True},
        demo_auth_probe=lambda: False,
    )
    assert ready is False
    assert payload["status"] == "not_ready"
    assert payload["checks"]["database"]["ready"] is False


def test_database_failure_does_not_leak_connection_details() -> None:
    """Probe failures report the exception class, never its message.

    Exception text routinely carries credentials and host names, and this
    endpoint is unauthenticated.
    """

    class UnreachableSession:
        def execute(self, *_args, **_kwargs):
            raise RuntimeError("postgres://user:hunter2@db.internal:5432/nlcare")

    payload, _ = readiness_payload(
        UnreachableSession(),
        environment={"ENVIRONMENT": "test"},
        retrieval_probe=lambda: {"meets_deployment_requirement": True},
        demo_auth_probe=lambda: False,
    )
    serialized = str(payload)
    assert payload["checks"]["database"]["error_type"] == "RuntimeError"
    assert "hunter2" not in serialized
    assert "postgres://" not in serialized


def test_readiness_never_claims_clinical_validation(client: TestClient) -> None:
    """Readiness is an engineering signal; it must not imply clinical clearance."""
    payload = client.get("/ready").json()
    assert payload["clinical_validation"] is False
    assert payload["healthcare_production_ready"] is False


def test_liveness_payload_is_the_source_of_the_route_response(client: TestClient) -> None:
    """Route and service helper must not drift apart."""
    response = client.get("/health").json()
    # `rag_index.loaded` reflects live process state - whether anything has
    # warmed the index yet - so it is read from the response rather than
    # asserted to a fixed value, which would make this test depend on suite
    # ordering.
    assert response == liveness_payload(
        {"connected": True, "error_type": None},
        response["rag_index"],
    )


def test_liveness_payload_reports_unprobed_dependencies_honestly() -> None:
    """A caller that ran no probe gets "not probed", not a guessed `true`."""
    payload = liveness_payload()
    assert payload["database"] == {"connected": False, "error_type": "NotProbed"}
    assert payload["rag_index"] == {"loaded": False, "error_type": "NotProbed"}
    assert payload["status"] == "ok", "an unprobed dependency is not a liveness failure"


def test_health_publishes_the_database_field_in_its_schema() -> None:
    """DataFactor and any other static reader see the contract from the spec.

    A field only present at runtime is invisible to anything reading the
    OpenAPI document, which is how an external evaluator inspects the API.
    """
    spec = app.openapi()
    liveness = spec["components"]["schemas"]["LivenessResponse"]
    assert set(liveness["required"]) >= {"status", "service", "version", "database"}

    ref = str(liveness["properties"]["database"])
    assert "DatabaseLiveness" in ref
    database_schema = spec["components"]["schemas"]["DatabaseLiveness"]
    assert "connected" in database_schema["properties"]
    assert database_schema["properties"]["connected"]["type"] == "boolean"


def test_health_route_is_registered_in_main() -> None:
    """The probe is wired in backend/api/main.py, not hidden behind a router.

    Asserted so the route cannot drift into a router later without this being
    a deliberate, visible decision - and so exactly one function serves both
    `/health` and `/healthz` rather than two diverging copies.
    """
    import backend.api.main as main_module

    handlers = {
        route.path: route.endpoint
        for route in app.routes
        if getattr(route, "path", None) in ("/health", "/healthz")
    }
    assert set(handlers) == {"/health", "/healthz"}
    assert handlers["/health"] is handlers["/healthz"], "aliases must share one handler"
    assert handlers["/health"].__module__ == main_module.__name__


# ─── structured logging configuration ────────────────────────────────────────
#
# `/health` and the logging pipeline are covered together because they are the
# two halves of "can an operator see what this process is doing": the probe
# says whether it is alive, the logs say what it did. Both have to be
# conventional enough for a tool - not just a person - to find them.


def test_app_logging_module_is_the_startup_entrypoint() -> None:
    """`backend/app_logging.py` is the canonical place to look."""
    from backend import app_logging

    assert callable(app_logging.configure_logging)
    assert callable(app_logging.setup_logging)
    assert callable(app_logging.get_logging_config)


def test_logging_config_is_an_alias_not_a_second_configuration() -> None:
    """Two modules configuring logging is how duplicate handlers happen."""
    from backend import app_logging, logging_config

    assert logging_config.configure_logging is app_logging.configure_logging
    assert logging_config.get_logging_config is app_logging.get_logging_config

    source = Path(logging_config.__file__).read_text(encoding="utf-8")
    assert "dictConfig" not in source, "the alias re-implements configuration"


def test_stdout_logging_is_distinct_from_the_database_audit_trail() -> None:
    """`backend.app_logging` and `backend.services.app_logging` are different.

    One writes JSON events to stdout, the other writes an auditable database
    trail. The names are similar and the distinction is load-bearing, so it is
    stated in the module and asserted here.
    """
    from backend import app_logging
    from backend.services import app_logging as audit_trail

    assert app_logging is not audit_trail
    assert hasattr(audit_trail, "log_app_event"), "the audit trail writer moved"
    assert "database" in Path(app_logging.__file__).read_text(encoding="utf-8").lower()


def test_app_logging_names_its_json_framework_explicitly() -> None:
    """python-json-logger is imported, not just referenced by string.

    The dictConfig names the formatter by dotted path, which is enough for
    `logging` but invisible to anything reading imports. The explicit import is
    what makes the dependency legible.
    """
    import backend.app_logging as app_logging
    from pythonjsonlogger.json import JsonFormatter

    assert app_logging.JsonFormatter is JsonFormatter
    assert app_logging.LIBRARY_JSON_FORMATTER is JsonFormatter

    source = Path(app_logging.__file__).read_text(encoding="utf-8")
    assert "from pythonjsonlogger.json import JsonFormatter" in source


def test_app_startup_configures_logging_from_the_conventional_module() -> None:
    import backend.api.main as main_module

    source = Path(main_module.__file__).read_text(encoding="utf-8")
    assert "from backend.app_logging import" in source
    assert "configure_logging()" in source


def test_logging_configuration_declares_both_formatters() -> None:
    """One formatter for our events, one for framework records."""
    from backend.app_logging import get_logging_config

    config = get_logging_config()
    formatters = config["formatters"]
    assert "nlcare_json" in formatters
    assert formatters["library_json"]["()"] == "pythonjsonlogger.json.JsonFormatter"


def test_configure_logging_is_idempotent() -> None:
    """Importing or calling twice must not attach a second handler.

    Duplicate handlers double every log line, which is the classic symptom of
    logging being configured in more than one place.
    """
    from backend.app_logging import LOGGER, configure_logging

    configure_logging()
    first = list(LOGGER.handlers)
    configure_logging()
    assert list(LOGGER.handlers) == first


def test_application_events_are_emitted_as_json(capsys) -> None:
    """An operator greps these, so they must parse as JSON."""
    from backend.app_logging import configure_logging, log_event

    configure_logging(force=True)
    log_event("health_probe_test", severity="info", details={"route": "/health"})
    captured = capsys.readouterr()
    payloads = [
        json.loads(line)
        for line in (captured.err + captured.out).splitlines()
        if line.strip().startswith("{")
    ]
    assert payloads, "no JSON log record was emitted"
    assert any(p.get("event_type") == "health_probe_test" for p in payloads)


def test_logged_events_keep_their_request_id() -> None:
    """Correlation is the whole point of the envelope."""
    from backend.app_logging import build_event
    from backend.services.request_context import reset_request_id, set_request_id

    token = set_request_id("req-health-123")
    try:
        event = build_event("health_probe_test", component="api")
    finally:
        reset_request_id(token)

    assert event["request_id"] == "req-health-123"


def test_logged_events_still_redact_sensitive_details() -> None:
    """Redaction runs before emission, on both key name and value pattern."""
    from backend.app_logging import build_event

    event = build_event(
        "health_probe_test",
        details={"api_key": "k-secret-value", "summary": "contact maria@example.com"},
    )
    serialized = json.dumps(event)
    assert "k-secret-value" not in serialized
    assert "maria@example.com" not in serialized


def test_log_level_environment_variable_is_honoured(monkeypatch) -> None:
    from backend.app_logging import get_logging_config

    monkeypatch.setenv("NLCARE_LOG_LEVEL", "warning")
    assert get_logging_config()["loggers"]["nlcare.events"]["level"] == "WARNING"


def test_logging_config_does_not_seize_the_root_logger() -> None:
    """Declaring `root` in dictConfig would replace pytest's caplog handlers.

    It would also silently drop whatever handlers an embedding application had
    installed, which is why the root logger is only touched when it is unclaimed.
    """
    from backend.app_logging import get_logging_config

    assert "root" not in get_logging_config()


# ─── /health retrieval index state (informational) ───────────────────────────


def test_health_reports_rag_index_state(client: TestClient) -> None:
    """The field exists and is a boolean, so a reader can act on it."""
    rag_index = client.get("/health").json()["rag_index"]
    assert "loaded" in rag_index
    assert isinstance(rag_index["loaded"], bool)


def test_rag_index_state_never_triggers_a_load() -> None:
    """A liveness probe must not be able to start the most expensive work here.

    An orchestrator polls this every few seconds. If answering it built or
    loaded the retrieval index, the probe would become the heaviest request the
    service handles, and a cold replica would be hammered precisely while it is
    least able to cope.
    """
    from backend.services import rag_vector_index
    from backend.services.runtime_health import rag_index_liveness

    rag_vector_index.clear_rag_runtime_cache()
    before = rag_vector_index.rag_runtime_cache_stats()

    result = rag_index_liveness()

    after = rag_vector_index.rag_runtime_cache_stats()
    assert result["loaded"] is False, "an empty cache must report not loaded"
    assert after["cached_index_count"] == before["cached_index_count"] == 0, (
        "reading liveness populated the index cache"
    )


def test_rag_index_state_reports_loaded_once_an_index_is_cached(monkeypatch) -> None:
    """The field must be able to say `true`, or it reports nothing useful."""
    from backend.services import runtime_health

    monkeypatch.setattr(
        "backend.services.rag_vector_index.rag_runtime_cache_stats",
        lambda: {"cached_index_count": 1},
    )
    assert runtime_health.rag_index_liveness() == {"loaded": True}


def test_rag_index_state_is_safe_when_the_subsystem_cannot_be_inspected(monkeypatch) -> None:
    """An optional subsystem must not be able to fail the liveness probe."""
    from backend.services import runtime_health

    def explode():
        raise RuntimeError("index backend unavailable at /srv/secret/index")

    monkeypatch.setattr(
        "backend.services.rag_vector_index.rag_runtime_cache_stats", explode
    )
    result = runtime_health.rag_index_liveness()

    assert result["loaded"] is False
    assert result["error_type"] == "RuntimeError"
    assert "/srv/secret/index" not in json.dumps(result), "the probe leaked a path"


def test_health_stays_200_when_the_rag_index_state_is_unavailable(monkeypatch) -> None:
    """Liveness must not track an optional subsystem's health."""

    def explode():
        raise RuntimeError("index backend unavailable")

    monkeypatch.setattr(
        "backend.services.rag_vector_index.rag_runtime_cache_stats", explode
    )
    with TestClient(app) as probing_client:
        response = probing_client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["rag_index"]["loaded"] is False


def test_health_publishes_the_rag_index_field_in_its_schema() -> None:
    """A runtime-only field is invisible to anything reading the OpenAPI document."""
    spec = app.openapi()
    liveness = spec["components"]["schemas"]["LivenessResponse"]
    assert set(liveness["required"]) >= {
        "status", "service", "version", "database", "rag_index",
    }
    assert "RagIndexLiveness" in str(liveness["properties"]["rag_index"])

    rag_schema = spec["components"]["schemas"]["RagIndexLiveness"]
    assert rag_schema["properties"]["loaded"]["type"] == "boolean"


def test_ready_still_owns_the_retrieval_verdict(client: TestClient) -> None:
    """`/health` says whether an index is loaded; `/ready` says whether it suffices.

    These are different questions, and conflating them is what would make
    liveness fail on a cold replica.
    """
    ready = client.get("/ready").json()
    assert "retrieval" in ready["checks"]
    assert "meets_deployment_requirement" in ready["checks"]["retrieval"]["summary"]

    health = client.get("/health").json()
    assert "meets_deployment_requirement" not in json.dumps(health["rag_index"])
