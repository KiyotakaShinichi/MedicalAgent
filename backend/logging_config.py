"""Alias for :mod:`backend.app_logging`, which is the canonical entrypoint.

Kept so ``from backend.logging_config import configure_logging`` continues to
work. It re-exports rather than re-implements: two modules configuring logging
is how duplicate handlers and double-emitted lines happen, so there is exactly
one configuration and this is a second name for it.

See :mod:`backend.app_logging` for the pipeline, the redaction policy, and the
distinction from :mod:`backend.services.app_logging` (which writes the database
audit trail rather than stdout).
"""

from __future__ import annotations

from backend.app_logging import (
    LIBRARY_JSON_FORMATTER,
    LIBRARY_LOG_FORMAT,
    LOGGER,
    JsonEventFormatter,
    JsonFormatter,
    build_event,
    configure_logging,
    get_logging_config,
    log_event,
    logging_config,
    new_correlation_id,
    setup_logging,
)

__all__ = [
    "LIBRARY_JSON_FORMATTER",
    "LIBRARY_LOG_FORMAT",
    "LOGGER",
    "JsonEventFormatter",
    "JsonFormatter",
    "build_event",
    "configure_logging",
    "get_logging_config",
    "log_event",
    "logging_config",
    "new_correlation_id",
    "setup_logging",
]
