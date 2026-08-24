"""Alias for :mod:`backend.logging_config`, which is the canonical module.

Kept so ``from backend.app_logging import configure_logging`` continues to
work. It re-exports rather than re-implements: two modules configuring logging
is how duplicate handlers and double-emitted lines happen, so there is exactly
one configuration and this is a second name for it.

Note the similarly-named :mod:`backend.services.app_logging`, which is a
different thing entirely - it writes the auditable database trail
(``AppEventLog``) rather than JSON to stdout. The distinction is stdout versus
database, and it is load-bearing.
"""

from __future__ import annotations

from backend.logging_config import (
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
