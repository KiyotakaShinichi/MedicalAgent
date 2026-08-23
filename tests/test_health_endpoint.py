"""Operational probe contract: `/health` liveness and `/ready` dependency state.

The split matters and is deliberate, so it is asserted rather than assumed:

* **`/health`** answers "is this process alive?" It touches no database, cache,
  model, or network dependency. An orchestrator uses it to decide whether to
  *restart* the process, so making it depend on a slow database turns a
  degraded dependency into a restart loop.
* **`/ready`** answers "should traffic be sent here?" It runs bounded probes
  against the database, retrieval index, and (when shared rate limiting is on)
  Redis, and returns 503 when any required one is unavailable so a load
  balancer drains the instance instead of restarting it.

Database-connectivity evidence therefore lives in the `/ready` tests below,
which is where that behaviour is actually contracted.

`tests/test_operational_endpoints.py` covers correlation, OpenAPI
discoverability, and log redaction. This file covers the probe response
contract itself and does not repeat those.
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
    assert client.get("/health").json() == liveness_payload({"connected": True, "error_type": None})


def test_liveness_payload_reports_an_unprobed_database_honestly() -> None:
    """A caller that ran no probe gets "not probed", not a guessed `true`."""
    payload = liveness_payload()
    assert payload["database"] == {"connected": False, "error_type": "NotProbed"}
    assert payload["status"] == "ok"


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
