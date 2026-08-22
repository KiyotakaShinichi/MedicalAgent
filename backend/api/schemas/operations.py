"""Response models for the operational probe endpoints.

These exist so `/health` and `/ready` carry a declared contract in the OpenAPI
document rather than an empty `schema: {}`. The endpoints and their payloads
predate this module; nothing here changes what they report, it only describes
it. Field sets mirror `backend.services.runtime_health` exactly - a
`response_model` filters the response, so an omission here would silently drop
a field that operators already depend on.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LivenessResponse(BaseModel):
    """Process-liveness answer. Deliberately does not touch dependencies."""

    status: str = Field(
        description="`ok` whenever the process can serve requests.",
        examples=["ok"],
    )
    service: str = Field(
        description="Stable service identifier.",
        examples=["nlcare_monitoring_prototype"],
    )


class ReadinessResponse(BaseModel):
    """Dependency-readiness answer, returned with 200 (ready) or 503 (not).

    `checks` is intentionally typed loosely: each probe contributes its own
    shape, and pinning it would freeze subsystem internals into the public
    contract. Probe failures report `error_type` only - never the exception
    message, which can carry connection strings or file paths.
    """

    status: str = Field(
        description="`ready` when every required probe answered, else `not_ready`.",
        examples=["ready"],
    )
    service: str = Field(examples=["nlcare_monitoring_prototype"])
    environment: str = Field(
        description="Deployment environment name, lowercased.",
        examples=["test"],
    )
    demo_auth_allowed: bool = Field(
        description="Whether demo authentication is currently permitted.",
    )
    checks: dict[str, Any] = Field(
        description=(
            "Per-dependency probe results keyed by subsystem "
            "(`database`, `retrieval`, `redis`). Each carries `ready`, and on "
            "failure an `error_type` class name with no message text."
        ),
    )
    clinical_validation: bool = Field(
        description="Always false. Readiness is an engineering signal only.",
    )
    healthcare_production_ready: bool = Field(
        description="Always false. This prototype is not cleared for patient care.",
    )
    claim_boundary: str = Field(
        description="Explicit statement of what readiness does and does not mean.",
    )


class ErrorResponse(BaseModel):
    """Body returned for an unhandled server error.

    Carries the correlation id so a report can be traced to a log line, and
    nothing else: no exception message, traceback, module path, or request
    body. `error` is a stable classification, not a detail string.
    """

    error: str = Field(
        description="Stable error classification, safe to display.",
        examples=["internal_server_error"],
    )
    request_id: str = Field(
        description="Correlation id, matching the `X-Request-ID` response header.",
    )
    detail: str = Field(
        description="Fixed, non-sensitive guidance. Never contains exception text.",
    )


__all__ = ["ErrorResponse", "LivenessResponse", "ReadinessResponse"]
