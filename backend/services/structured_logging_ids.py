"""Dependency-free identifier helpers for logging and request context."""

from __future__ import annotations

import uuid


def new_request_id(prefix: str = "req") -> str:
    """Return an opaque, bounded identifier suitable for headers and logs."""
    safe_prefix = prefix if prefix.isalnum() and len(prefix) <= 12 else "req"
    return f"{safe_prefix}_{uuid.uuid4().hex[:16]}"


__all__ = ["new_request_id"]
