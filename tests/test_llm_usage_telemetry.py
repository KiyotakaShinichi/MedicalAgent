from backend.services.agent_eval_scoring import estimate_token_and_cost
from backend.services.llm_telemetry import (
    LLMCallTimer,
    record_llm_call,
    reset_llm_telemetry,
    snapshot_llm_telemetry,
    start_llm_telemetry,
)


def test_provider_reported_usage_is_not_mixed_with_estimate():
    token = start_llm_telemetry()
    try:
        record_llm_call(
            provider="groq",
            model="test-model",
            operation="patient_support_answer",
            latency_ms=123.4,
            prompt_parts=["long prompt that should not affect actual usage"],
            completion_text="long completion",
            usage={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
        )
        report = snapshot_llm_telemetry()
    finally:
        reset_llm_telemetry(token)

    assert report["provider_reported_call_count"] == 1
    assert report["estimated_call_count"] == 0
    assert report["input_tokens"] == 11
    assert report["output_tokens"] == 7
    assert report["total_tokens"] == 18
    assert report["actual_usage_coverage_rate"] == 1.0
    assert report["content_retained"] is False
    assert "prompt" not in report["calls"][0]
    assert "completion" not in report["calls"][0]


def test_missing_provider_usage_is_explicit_estimate():
    token = start_llm_telemetry()
    try:
        record_llm_call(
            provider="ollama",
            model="local-test",
            operation="structured_router",
            latency_ms=20,
            prompt_parts=["12345678"],
            completion_text="1234",
        )
        report = snapshot_llm_telemetry()
    finally:
        reset_llm_telemetry(token)

    assert report["provider_reported_call_count"] == 0
    assert report["estimated_call_count"] == 1
    assert report["calls"][0]["usage_basis"] == "chars_div_4_estimate"
    assert report["actual_usage_coverage_rate"] == 0.0


def test_failed_call_is_counted_without_error_message_content():
    token = start_llm_telemetry()
    try:
        record_llm_call(
            provider="groq",
            model="test-model",
            operation="structured_router",
            latency_ms=5,
            prompt_parts=["private prompt"],
            success=False,
            error_type="TimeoutError",
        )
        report = snapshot_llm_telemetry()
    finally:
        reset_llm_telemetry(token)

    assert report["failed_call_count"] == 1
    assert report["calls"][0]["error_type"] == "TimeoutError"
    assert "private prompt" not in str(report)


def test_pipeline_estimate_keeps_provider_usage_separate():
    usage = {
        "call_count": 1,
        "provider_reported_call_count": 1,
        "estimated_call_count": 0,
        "input_tokens": 100,
        "output_tokens": 25,
        "total_tokens": 125,
        "estimated_cost_usd": 0.00003,
        "content_retained": False,
    }
    result = estimate_token_and_cost(
        "short query",
        "short answer",
        [{"text": "retrieved context"}],
        usage,
    )

    assert result["provider_usage_captured"] is True
    assert result["provider_token_usage"]["total_tokens"] == 125
    assert result["estimated_total_tokens"] != 125
    assert result["estimated_llm_cost_usd"] == 0.00003


def test_timer_returns_non_negative_latency():
    timer = LLMCallTimer.start()
    assert timer.elapsed_ms() >= 0
