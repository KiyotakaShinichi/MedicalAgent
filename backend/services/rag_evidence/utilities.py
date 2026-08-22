"""Value coercion, hashing, and request-context helpers."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence


def response_digest(reply: Any) -> str:
    return hashlib.sha256(str(reply or "").encode("utf-8")).hexdigest()


def current_request_id() -> str | None:
    try:
        from backend.services.request_context import get_request_id

        return get_request_id()
    except Exception:  # noqa: BLE001
        return None


def coerce_chunks(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def chunk_id(chunk: Mapping[str, Any]) -> str:
    return str(chunk.get("id") or chunk.get("chunk_id") or "")


def dedupe_codes(values: Sequence[str] | None) -> list[str]:
    return list(dict.fromkeys(str(value)[:160] for value in (values or []) if value))


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def safe_float(value: Any) -> float:
    try:
        return round(float(value or 0), 4)
    except (TypeError, ValueError):
        return 0.0
