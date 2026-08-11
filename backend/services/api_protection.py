"""Bounded API abuse controls for local and shared synthetic deployments."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from hashlib import sha256

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


class RedisRateLimiter:
    """Shared fixed-window limiter for multi-process synthetic deployments."""

    def __init__(self, redis_url: str) -> None:
        from redis import Redis

        self.client = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )

    def check(
        self,
        key: str,
        *,
        limit: int,
        window_seconds: int,
        now: float | None = None,
    ) -> tuple[bool, int, int]:
        current = time.time() if now is None else now
        bucket = int(current // window_seconds)
        digest = sha256(key.encode("utf-8")).hexdigest()
        redis_key = f"nlcare:rate:{bucket}:{digest}"
        pipeline = self.client.pipeline(transaction=True)
        pipeline.incr(redis_key)
        pipeline.expire(redis_key, window_seconds + 5)
        count, _ = pipeline.execute()
        remaining = max(0, limit - int(count))
        retry_after = max(1, int(((bucket + 1) * window_seconds) - current))
        return int(count) <= limit, remaining, retry_after


class EngineeringApiProtectionMiddleware(BaseHTTPMiddleware):
    """Add bounded, observable controls without claiming production security."""

    def __init__(self, app) -> None:  # noqa: ANN001
        super().__init__(app)
        self.max_request_bytes = int(os.environ.get("NLCARE_MAX_REQUEST_BYTES", DEFAULT_MAX_REQUEST_BYTES))
        self.window_seconds = int(os.environ.get("NLCARE_RATE_WINDOW_SECONDS", "60"))
        self.local_limiter = ProcessLocalRateLimiter()
        self.shared_enabled = os.environ.get("NLCARE_SHARED_RATE_LIMIT_ENABLED", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        redis_url = str(os.environ.get("REDIS_URL") or "").strip()
        self.shared_limiter = RedisRateLimiter(redis_url) if self.shared_enabled and redis_url else None
        self.strict_profile = str(
            os.environ.get("ENVIRONMENT") or os.environ.get("APP_ENV") or "development"
        ).lower() in {"staging", "production", "prod"}

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
            organization_id = request.headers.get("x-nlcare-organization-id", "unselected")
            limit_key = f"{organization_id}:{client_host}:{request.method}:{request.url.path}"
            if self.shared_enabled and self.shared_limiter is None and self.strict_profile:
                response = _error(503, "Shared API protection is not configured; request failed closed.")
                response.headers["X-RateLimit-Scope"] = "shared_redis_missing_fail_closed"
                return response
            limiter = self.shared_limiter or self.local_limiter
            try:
                allowed, remaining, retry_after = limiter.check(
                    limit_key,
                    limit=limit,
                    window_seconds=self.window_seconds,
                )
            except Exception:
                if self.strict_profile:
                    response = _error(503, "Shared API protection is unavailable; request failed closed.")
                    response.headers["X-RateLimit-Scope"] = "shared_redis_unavailable_fail_closed"
                    return response
                allowed, remaining, retry_after = self.local_limiter.check(
                    limit_key,
                    limit=limit,
                    window_seconds=self.window_seconds,
                )
            if not allowed:
                response = _error(429, "Too many requests for this prototype endpoint. Please retry later.")
                response.headers["Retry-After"] = str(retry_after)
                response.headers["X-RateLimit-Limit"] = str(limit)
                response.headers["X-RateLimit-Remaining"] = "0"
                response.headers["X-RateLimit-Scope"] = (
                    "shared_redis_synthetic_control" if self.shared_limiter else "process_local_engineering_control"
                )
                return response

        response = await call_next(request)
        response.headers.setdefault(
            "X-API-Protection-Scope",
            "shared_redis_synthetic_control" if self.shared_limiter else "process_local_engineering_control",
        )
        return response


def _route_limit(method: str, path: str) -> int | None:
    if method == "POST" and path in {"/auth/demo-login", "/auth/demo-credential-login"}:
        return int(os.environ.get("NLCARE_AUTH_RATE_LIMIT", "120"))
    if method == "POST" and path in {"/me/chat", "/me/chat/stream"}:
        return int(os.environ.get("NLCARE_CHAT_RATE_LIMIT", "120"))
    if method == "POST" and path.startswith("/me/uploads"):
        return int(os.environ.get("NLCARE_UPLOAD_RATE_LIMIT", "30"))
    if method == "POST" and path.startswith("/platform/organizations") and path.endswith("/jobs"):
        return int(os.environ.get("NLCARE_PLATFORM_JOB_RATE_LIMIT", "60"))
    if method == "POST" and path.startswith("/platform/organizations"):
        return int(os.environ.get("NLCARE_PLATFORM_WRITE_RATE_LIMIT", "30"))
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
    "RedisRateLimiter",
]
