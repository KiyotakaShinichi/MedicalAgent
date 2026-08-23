"""Application logging configuration entrypoint.

This is the conventional place to look for how NLCare configures logging, and
the module the FastAPI app imports at startup. The implementation lives in
:mod:`backend.services.structured_logging`, which already owns the redaction
policy and the request-id correlation; this module is the front door to it, not
a second logging system.

Structured JSON logging is provided by **python-json-logger**, imported
explicitly below. Two formatters share one stdlib `logging.config.dictConfig`
pipeline:

* application events carry the redacted ``nlcare_event`` envelope - request id,
  correlation id, severity, sanitised details;
* framework and third-party records (uvicorn, sqlalchemy, httpx) have no such
  envelope, so `JsonFormatter` renders them as the structured records they
  already are.

Behaviour this module deliberately preserves, because it is load-bearing:

* ``NLCARE_LOG_LEVEL`` / ``NLCARE_ROOT_LOG_LEVEL`` still select levels;
* PII, API-key, and secret redaction still runs before anything is emitted;
* the request id set by the API middleware still reaches every event;
* ``configure_logging`` stays idempotent, so importing this module twice - or
  importing it after the app already configured logging - does not attach a
  second handler or reinitialise anything;
* the root logger is only touched when it has no handlers of its own, which is
  what keeps pytest's ``caplog`` capture working.

Importing this module has no side effects. Configuration happens when
``configure_logging()`` is called, which the app does once at startup.
"""

from __future__ import annotations

import logging
from typing import Any

# The structured-logging framework, imported explicitly so both a reader and a
# static analyser can see which one this application uses. The dictConfig below
# also names it by path, but a string in a config dict is not an import.
from pythonjsonlogger.json import JsonFormatter

from backend.services.structured_logging import (
    JsonEventFormatter,
    build_event,
    configure_logging,
    log_event,
    logging_config,
    new_correlation_id,
)

#: The formatter used for framework and third-party log records.
LIBRARY_JSON_FORMATTER = JsonFormatter

#: Format string applied to those records.
LIBRARY_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

#: The application event logger. Everything routed through :func:`log_event`
#: lands here already redacted.
LOGGER = logging.getLogger("nlcare.events")


def get_logging_config() -> dict[str, Any]:
    """The `dictConfig` schema this application logs under.

    Exposed as data so it can be asserted on in tests and read by anyone
    auditing how logs are produced, rather than being implied by a sequence of
    ``addHandler`` calls.
    """
    return logging_config()


def setup_logging(*, force: bool = False) -> None:
    """Install the JSON logging pipeline. Idempotent.

    Named for the convention; :func:`configure_logging` is the same function
    and remains available for existing callers.
    """
    configure_logging(force=force)


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
