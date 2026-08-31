"""Vendor-neutral, failure-isolated runtime error reporting.

Core NLCare emits redacted structured events and requires no hosted monitoring
provider. A deployment can install another :class:`ErrorReporter` at startup;
call sites remain unchanged and a broken reporter can never break a request.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable

from backend.services.structured_logging import log_event


class OperationalErrorCategory(str, Enum):
    VALIDATION = "validation_error"
    AUTHENTICATION = "authentication_error"
    AUTHORIZATION = "authorization_error"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    CONFIGURATION = "configuration_error"
    STORAGE = "storage_error"
    PROVIDER = "provider_error"
    INTERNAL_SERVICE = "internal_service_error"


@runtime_checkable
class ErrorReporter(Protocol):
    def capture_exception(
        self,
        error: BaseException,
        *,
        category: OperationalErrorCategory,
        request_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None: ...

    def capture_message(
        self,
        message: str,
        *,
        category: OperationalErrorCategory,
        request_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None: ...


class NoOpErrorReporter:
    """Explicitly discard reports, useful for constrained embeddings."""

    def capture_exception(
        self,
        error: BaseException,
        *,
        category: OperationalErrorCategory,
        request_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        return None

    def capture_message(
        self,
        message: str,
        *,
        category: OperationalErrorCategory,
        request_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        return None


class StructuredLogErrorReporter:
    """Report operational failures through the canonical redacted logger."""

    def capture_exception(
        self,
        error: BaseException,
        *,
        category: OperationalErrorCategory,
        request_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        log_event(
            "runtime_exception",
            severity="error",
            request_id=request_id,
            component="runtime",
            details={
                **(context or {}),
                "error_category": category.value,
                "error_type": type(error).__name__,
            },
        )

    def capture_message(
        self,
        message: str,
        *,
        category: OperationalErrorCategory,
        request_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        # The caller's prose is deliberately not logged. A category and bounded
        # context are enough for operational routing without creating a new
        # patient-content disclosure surface.
        log_event(
            "runtime_message",
            severity="warning",
            request_id=request_id,
            component="runtime",
            details={**(context or {}), "error_category": category.value},
        )


_reporter: ErrorReporter = StructuredLogErrorReporter()


def get_error_reporter() -> ErrorReporter:
    return _reporter


def set_error_reporter(reporter: ErrorReporter) -> ErrorReporter:
    """Install an adapter and return the previous reporter for safe restoration."""
    global _reporter
    previous = _reporter
    _reporter = reporter
    return previous


def capture_exception(
    error: BaseException,
    *,
    category: OperationalErrorCategory = OperationalErrorCategory.INTERNAL_SERVICE,
    request_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    """Best-effort exception reporting; observability must never alter serving."""
    try:
        _reporter.capture_exception(
            error,
            category=category,
            request_id=request_id,
            context=context,
        )
    except Exception:
        return None


def capture_message(
    message: str,
    *,
    category: OperationalErrorCategory,
    request_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    try:
        _reporter.capture_message(
            message,
            category=category,
            request_id=request_id,
            context=context,
        )
    except Exception:
        return None


__all__ = [
    "ErrorReporter",
    "NoOpErrorReporter",
    "OperationalErrorCategory",
    "StructuredLogErrorReporter",
    "capture_exception",
    "capture_message",
    "get_error_reporter",
    "set_error_reporter",
]
