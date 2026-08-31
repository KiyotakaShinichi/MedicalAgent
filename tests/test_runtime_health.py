from __future__ import annotations

from backend.services.runtime_health import (
    application_version,
    database_connectivity,
    liveness_payload,
    readiness_payload,
)


class _Database:
    def __init__(self, *, fails: bool = False):
        self.fails = fails
        self.calls = 0

    def execute(self, _statement):
        self.calls += 1
        if self.fails:
            raise ConnectionError("database unavailable")
        return 1


def _retrieval_ready():
    return {"meets_deployment_requirement": True, "backend": "test_index"}


def test_liveness_reports_status_version_and_database():
    """The payload's shape, including the informational database field.

    `version` tells an operator which build answered; `database` makes
    reachability visible from the conventional probe. Neither is allowed to
    change `status` - see `test_liveness_verdict_ignores_the_database` below.
    """
    payload = liveness_payload(database_connectivity(_Database()))

    assert payload["status"] == "ok"
    assert payload["service"] == "nlcare_monitoring_prototype"
    assert payload["version"] == application_version()
    assert payload["database"] == {"connected": True}
    assert isinstance(payload["rag_index"]["loaded"], bool)


def test_liveness_verdict_ignores_the_database():
    """A live process with a dead database is still live.

    This is the property that keeps liveness from becoming a restart loop:
    restarting a process cannot repair a database, so the verdict must not
    track it. `/ready` owns the fail-closed decision.
    """
    payload = liveness_payload(database_connectivity(_Database(fails=True)))

    assert payload["status"] == "ok"
    assert payload["database"]["connected"] is False
    assert payload["database"]["error_type"] == "ConnectionError"
    assert "database unavailable" not in str(payload), "probe leaked the message"


def test_liveness_without_a_probe_is_reported_as_unprobed():
    """A caller that ran no probe gets "not probed", never a guessed `true`."""
    payload = liveness_payload()
    assert payload["database"] == {"connected": False, "error_type": "NotProbed"}
    assert payload["rag_index"] == {"loaded": False, "error_type": "NotProbed"}


def test_readiness_passes_when_required_dependencies_answer():
    db = _Database()

    payload, ready = readiness_payload(
        db,
        environment={"APP_ENV": "test"},
        retrieval_probe=_retrieval_ready,
        demo_auth_probe=lambda: True,
    )

    assert ready is True
    assert payload["status"] == "ready"
    assert payload["checks"]["database"] == {"ready": True, "required": True}
    assert payload["checks"]["retrieval"]["ready"] is True
    assert payload["checks"]["retrieval"]["required"] is True
    assert payload["demo_auth_allowed"] is True
    assert payload["clinical_validation"] is False
    assert payload["healthcare_production_ready"] is False


def test_readiness_fails_closed_when_database_is_unavailable():
    payload, ready = readiness_payload(
        _Database(fails=True),
        environment={"APP_ENV": "test"},
        retrieval_probe=_retrieval_ready,
    )

    assert ready is False
    assert payload["status"] == "not_ready"
    assert payload["checks"]["database"] == {
        "ready": False,
        "required": True,
        "error_type": "ConnectionError",
    }


def test_readiness_fails_closed_when_retrieval_probe_throws():
    def broken_retrieval():
        raise RuntimeError("index unavailable")

    payload, ready = readiness_payload(
        _Database(),
        environment={"APP_ENV": "test"},
        retrieval_probe=broken_retrieval,
    )

    assert ready is False
    assert payload["checks"]["retrieval"] == {
        "ready": False,
        "required": True,
        "error_type": "RuntimeError",
    }


def test_readiness_fails_closed_when_required_configuration_probe_throws():
    payload, ready = readiness_payload(
        _Database(),
        environment={"APP_ENV": "test"},
        retrieval_probe=_retrieval_ready,
        demo_auth_probe=lambda: (_ for _ in ()).throw(ValueError("bad secret")),
    )

    assert ready is False
    assert payload["demo_auth_allowed"] is False
    assert payload["checks"]["configuration"] == {
        "ready": False,
        "required": True,
        "error_type": "ValueError",
    }


def test_readiness_does_not_expose_unbounded_retrieval_details():
    payload, ready = readiness_payload(
        _Database(),
        environment={"APP_ENV": "test"},
        retrieval_probe=lambda: {
            "meets_deployment_requirement": False,
            "status": "missing",
            "path": "C:/private/patient/index.json",
            "source_url": "https://user:secret@example.invalid",
        },
    )

    summary = payload["checks"]["retrieval"]["summary"]
    assert ready is False
    assert summary == {"status": "missing", "meets_deployment_requirement": False}


def test_optional_external_provider_configuration_does_not_control_readiness():
    payload, ready = readiness_payload(
        _Database(),
        environment={
            "APP_ENV": "test",
            "GROQ_API_KEY": "configured-but-not-probed",
            "PINECONE_API_KEY": "configured-but-not-probed",
        },
        retrieval_probe=_retrieval_ready,
    )

    assert ready is True
    assert payload["status"] == "ready"
    assert set(payload["checks"]) == {"database", "retrieval", "redis"}


def test_readiness_requires_redis_url_when_shared_limit_is_enabled():
    payload, ready = readiness_payload(
        _Database(),
        environment={"APP_ENV": "test", "NLCARE_SHARED_RATE_LIMIT_ENABLED": "true"},
        retrieval_probe=_retrieval_ready,
    )

    assert ready is False
    assert payload["checks"]["redis"] == {
        "ready": False,
        "required": True,
        "error_type": "MissingRedisUrl",
    }
