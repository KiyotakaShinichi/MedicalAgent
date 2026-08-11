import asyncio

from starlette.requests import Request

from backend.services.api_protection import EngineeringApiProtectionMiddleware, ProcessLocalRateLimiter


def test_process_local_limiter_blocks_after_limit_and_resets() -> None:
    limiter = ProcessLocalRateLimiter()

    assert limiter.check("chat", limit=2, window_seconds=60, now=10.0)[0]
    assert limiter.check("chat", limit=2, window_seconds=60, now=11.0)[0]
    allowed, remaining, retry_after = limiter.check("chat", limit=2, window_seconds=60, now=12.0)
    assert not allowed
    assert remaining == 0
    assert retry_after > 0

    assert limiter.check("chat", limit=2, window_seconds=60, now=71.0)[0]


def test_process_local_limiter_isolates_keys() -> None:
    limiter = ProcessLocalRateLimiter()
    assert limiter.check("auth:a", limit=1, window_seconds=60, now=1.0)[0]
    assert not limiter.check("auth:a", limit=1, window_seconds=60, now=2.0)[0]
    assert limiter.check("auth:b", limit=1, window_seconds=60, now=2.0)[0]


def test_strict_shared_rate_limit_fails_closed_when_redis_is_not_configured(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "staging")
    monkeypatch.setenv("NLCARE_SHARED_RATE_LIMIT_ENABLED", "true")
    monkeypatch.delenv("REDIS_URL", raising=False)
    middleware = EngineeringApiProtectionMiddleware(lambda scope, receive, send: None)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "path": "/auth/demo-login",
            "raw_path": b"/auth/demo-login",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8017),
        }
    )

    async def should_not_run(_request):
        raise AssertionError("Strict missing shared protection must stop before the route")

    response = asyncio.run(middleware.dispatch(request, should_not_run))
    assert response.status_code == 503
    assert response.headers["X-RateLimit-Scope"] == "shared_redis_missing_fail_closed"
