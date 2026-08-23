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

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.main import app  # noqa: E402
from backend.services.runtime_health import (  # noqa: E402
    application_version,
    liveness_payload,
    readiness_payload,
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# ─── GET /health ─────────────────────────────────────────────────────────────


def test_health_returns_200(client: TestClient) -> None:
    assert client.get("/health").status_code == 200


def test_health_exposes_the_expected_keys(client: TestClient) -> None:
    """Exact key set: a probe consumer breaks on an unannounced field change."""
    assert set(client.get("/health").json()) == {"status", "service", "version"}


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


def test_health_does_not_touch_the_database(client: TestClient) -> None:
    """Liveness decides restarts, so it must not depend on a slow dependency.

    A database-backed liveness probe converts a degraded database into a
    restart loop. The payload reporting no dependency state is the observable
    proxy for that.
    """
    payload = client.get("/health").json()
    assert "checks" not in payload
    assert "database" not in str(payload).lower()


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
    assert client.get("/health").json() == liveness_payload()
