"""Process-local API abuse controls for the engineering prototype.

These controls bound obvious oversized requests and burst traffic. They are
deliberately labelled process-local: a real multi-instance deployment would
need a shared gateway or rate-limit store and an independent security review.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


DEFAULT_MAX_REQUEST_BYTES = 24 * 1024 * 1024


@dataclass
class _Window:
    started_at: float
    count: int


class ProcessLocalRateLimiter:
    def __init__(self) -> None:
        self._windows: dict[str, _Window] = {}
        self._lock = threading.Lock()

    def check(
        self,
        key: str,
        *,
        limit: int,
        window_seconds: int,
        now: float | None = None,
    ) -> tuple[bool, int, int]:
        current = time.monotonic() if now is None else now
        with self._lock:
            state = self._windows.get(key)
            if state is None or current - state.started_at >= window_seconds:
                state = _Window(started_at=current, count=0)
                self._windows[key] = state
            state.count += 1
            remaining = max(0, limit - state.count)
            retry_after = max(1, int(window_seconds - (current - state.started_at)))
            return state.count <= limit, remaining, retry_after


class EngineeringApiProtectionMiddleware(BaseHTTPMiddleware):
    """Add bounded, observable controls without claiming production security."""

    def __init__(self, app) -> None:  # noqa: ANN001
        super().__init__(app)
        self.max_request_bytes = int(os.environ.get("NLCARE_MAX_REQUEST_BYTES", DEFAULT_MAX_REQUEST_BYTES))
        self.window_seconds = int(os.environ.get("NLCARE_RATE_WINDOW_SECONDS", "60"))
        self.limiter = ProcessLocalRateLimiter()

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                request_bytes = int(content_length)
            except ValueError:
                return _error(400, "Invalid Content-Length header.")
            if request_bytes > self.max_request_bytes:
                response = _error(413, "Request exceeds the engineering prototype upload limit.")
                response.headers["X-NLCare-Request-Limit"] = str(self.max_request_bytes)
                return response

        limit = _route_limit(request.method, request.url.path)
        if limit is not None:
            client_host = request.client.host if request.client else "unknown"
            allowed, remaining, retry_after = self.limiter.check(
                f"{client_host}:{request.method}:{request.url.path}",
                limit=limit,
                window_seconds=self.window_seconds,
            )
            if not allowed:
                response = _error(429, "Too many requests for this prototype endpoint. Please retry later.")
                response.headers["Retry-After"] = str(retry_after)
                response.headers["X-RateLimit-Limit"] = str(limit)
                response.headers["X-RateLimit-Remaining"] = "0"
                response.headers["X-RateLimit-Scope"] = "process_local_engineering_control"
                return response

        response = await call_next(request)
        response.headers.setdefault("X-API-Protection-Scope", "process_local_engineering_control")
        return response


def _route_limit(method: str, path: str) -> int | None:
    if method == "POST" and path in {"/auth/demo-login", "/auth/demo-credential-login"}:
        return int(os.environ.get("NLCARE_AUTH_RATE_LIMIT", "120"))
    if method == "POST" and path in {"/me/chat", "/me/chat/stream"}:
        return int(os.environ.get("NLCARE_CHAT_RATE_LIMIT", "120"))
    if method == "POST" and path.startswith("/me/uploads"):
        return int(os.environ.get("NLCARE_UPLOAD_RATE_LIMIT", "30"))
    return None


def _error(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "detail": detail,
            "clinical_validation": False,
            "healthcare_production_ready": False,
        },
    )


__all__ = [
    "DEFAULT_MAX_REQUEST_BYTES",
    "EngineeringApiProtectionMiddleware",
    "ProcessLocalRateLimiter",
]
