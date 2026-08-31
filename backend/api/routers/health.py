"""Operational probe routes: `GET /health` and `GET /ready`.

Two endpoints answering two different questions. Conflating them is the mistake
this split exists to avoid, and both halves are load-bearing:

* **`/health`** is a *liveness* probe. It returns 200 whenever the process can
  serve requests, and reports `status`, `service`, `version`, database
  reachability, and whether the retrieval index is loaded here. The dependency
  fields are **informational**: they change the field and nothing else, not
  `status` and not the HTTP code.
* **`/ready`** is the authoritative *readiness* probe. It aggregates database,
  retrieval, and — when shared rate limiting is on — Redis, and returns **503**
  when any required dependency is unavailable.

Why `/health` does not return 503 when the database is down
-----------------------------------------------------------
An orchestrator uses liveness to decide whether to **restart** the process, and
restarting cannot repair a database. A liveness probe that fails on a
dependency outage converts that outage into a cluster-wide restart loop, taking
down replicas that were serving fine. Draining traffic is the correct response,
and that is what `/ready`'s 503 is for.

The database probe is also bounded, because a liveness probe that *hangs* is a
restart vector too: an orchestrator that times out waiting restarts the process
just as surely as a 500 would. The retrieval field is read from in-process
cache counters and never loads or builds an index — answering a probe polled
every few seconds must not trigger the most expensive work the service does.

Neither route is authenticated, so a failing probe reports an exception *class
name* at most. Exception messages routinely carry connection strings.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.api.schemas.operations import LivenessResponse, ReadinessResponse
from backend.services.runtime_health import (
    database_connectivity,
    liveness_payload,
    rag_index_liveness,
    readiness_payload,
)
from backend.services.runtime_metrics import record_readiness
from backend.services.structured_logging import log_event

router = APIRouter(tags=["operations"])


@router.get("/health", summary="Liveness probe", response_model=LivenessResponse,
            operation_id="getHealth",
            responses={200: {"description": "Process is alive and able to serve requests."}})
@router.get("/healthz", include_in_schema=False, response_model=LivenessResponse)
def health(db: Session = Depends(get_db)) -> dict:
    """Liveness probe. Returns 200 whenever the process can serve requests.

    Reports `status`, `service`, `version`, database reachability, and whether
    the retrieval index is loaded in this process. The dependency fields are
    informational — see the module docstring for why a dead database still
    returns 200 here and 503 on `/ready`.

    `/healthz` is an unlisted alias for orchestrators probing the
    Kubernetes-conventional path. It shares this handler, so the two answers
    cannot drift apart.
    """
    return liveness_payload(database_connectivity(db), rag_index_liveness())


@router.get(
    "/ready",
    summary="Readiness probe",
    response_model=ReadinessResponse,
    operation_id="getReady",
    responses={
        200: {"description": "Every required dependency answered its probe."},
        503: {
            "description": "At least one required dependency is not ready.",
            "model": ReadinessResponse,
        },
    },
)
@router.get("/readyz", include_in_schema=False, response_model=ReadinessResponse)
def ready(response: Response, db: Session = Depends(get_db)) -> dict:
    """Runtime readiness probe for engineering deployments.

    Checks database reachability, retrieval-index availability, and — only when
    shared rate limiting is enabled — Redis. Each probe is bounded, and a
    failing probe reports its exception *class name* only, never the message,
    which can carry connection strings or filesystem paths.

    Returns 503 when any required dependency is not ready, so a load balancer
    can drain the instance without restarting it.

    This reports deployment posture. It does not imply healthcare production
    readiness or clinical validation.

    `/readyz` is an unlisted alias for the Kubernetes-conventional path.
    """
    from backend.services.auth import is_demo_auth_allowed
    from backend.services.rag_vector_index import rag_runtime_readiness

    payload, is_ready = readiness_payload(
        db,
        retrieval_probe=rag_runtime_readiness,
        demo_auth_probe=is_demo_auth_allowed,
    )
    record_readiness(ready=is_ready)
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        failed_checks = sorted(
            name for name, check in payload["checks"].items() if not check.get("ready")
        )
        log_event(
            "readiness_probe_failed",
            severity="warning",
            component="operations",
            details={"failed_checks": failed_checks, "status_code": 503},
        )
    return payload


__all__ = ["health", "ready", "router"]
