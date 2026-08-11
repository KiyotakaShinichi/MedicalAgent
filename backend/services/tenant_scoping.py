"""Deterministic tenant-scoped identifiers for shared infrastructure."""

from __future__ import annotations

import hashlib
import re
from typing import Any


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _component(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not _SAFE_COMPONENT.fullmatch(text):
        raise ValueError(f"Invalid {label} for tenant-scoped infrastructure key.")
    return text


def tenant_cache_key(
    organization_id: str,
    project_id: str,
    *parts: Any,
    environment_id: str | None = None,
) -> str:
    """Return a non-ambiguous cache key with explicit tenant dimensions."""

    organization = _component(organization_id, "organization_id")
    project = _component(project_id, "project_id")
    environment = _component(environment_id or "default", "environment_id")
    suffix = ":".join(_component(part, "cache key component") for part in parts) or "root"
    return f"nlcare:v1:org:{organization}:project:{project}:env:{environment}:{suffix}"


def tenant_vector_namespace(
    organization_id: str,
    project_id: str,
    *,
    environment_id: str | None = None,
) -> str:
    """Return a stable opaque vector namespace for one tenant environment."""

    raw = tenant_cache_key(
        organization_id,
        project_id,
        "vectors",
        environment_id=environment_id,
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"nlcare_{digest}"


__all__ = ["tenant_cache_key", "tenant_vector_namespace"]
