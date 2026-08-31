"""Bounded request-correlation context shared across runtime services."""

from __future__ import annotations

import re
from contextvars import ContextVar, Token

from backend.services.structured_logging_ids import new_request_id


MAX_REQUEST_ID_LENGTH = 128
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_REQUEST_ID: ContextVar[str | None] = ContextVar("request_id", default=None)


def normalize_request_id(candidate: str | None) -> str:
    """Preserve a safe caller ID, otherwise mint a bounded server ID.

    Request IDs are reflected in response headers and logs. Accepting arbitrary
    header text would therefore create a log-injection and high-cardinality
    surface. The accepted grammar is intentionally small but covers UUIDs and
    the IDs commonly emitted by gateways and tracing proxies.
    """
    value = candidate.strip() if isinstance(candidate, str) else ""
    if len(value) <= MAX_REQUEST_ID_LENGTH and _VALID_REQUEST_ID.fullmatch(value):
        return value
    return new_request_id()


def set_request_id(request_id: str) -> Token[str | None]:
    return _REQUEST_ID.set(request_id)


def get_request_id() -> str | None:
    return _REQUEST_ID.get()


def reset_request_id(token: Token[str | None]) -> None:
    _REQUEST_ID.reset(token)


__all__ = [
    "MAX_REQUEST_ID_LENGTH",
    "get_request_id",
    "normalize_request_id",
    "reset_request_id",
    "set_request_id",
]
