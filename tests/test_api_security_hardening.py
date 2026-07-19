from backend.services.api_protection import ProcessLocalRateLimiter


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
