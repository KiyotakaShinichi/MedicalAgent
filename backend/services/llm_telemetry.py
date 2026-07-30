"""Request-scoped LLM usage telemetry without prompt or response retention.

Provider-reported token counts are kept separate from local estimates. Dollar
figures use explicit engineering pricing assumptions and are not billing truth.
"""
from __future__ import annotations

import os
from contextvars import ContextVar, Token
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable


CLAIM_BOUNDARY = (
    "Token, latency, and cost fields are engineering telemetry. Cost uses "
    "configurable pricing assumptions, not audited provider billing, and none "
    "of these measurements establish clinical validation or patient benefit."
)

_INPUT_COST_PER_MILLION_USD = float(
    os.environ.get("NLCARE_LLM_INPUT_COST_PER_MILLION_USD", "0.15")
)
_OUTPUT_COST_PER_MILLION_USD = float(
    os.environ.get("NLCARE_LLM_OUTPUT_COST_PER_MILLION_USD", "0.60")
)

_CALLS: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "nlcare_llm_telemetry_calls",
    default=None,
)


@dataclass(frozen=True)
class LLMCallTimer:
    started: float

    @classmethod
    def start(cls) -> "LLMCallTimer":
        return cls(started=perf_counter())

    def elapsed_ms(self) -> float:
        return round((perf_counter() - self.started) * 1000.0, 2)


def start_llm_telemetry() -> Token:
    """Start an isolated collector for one request or direct pipeline run."""
    return _CALLS.set([])


def reset_llm_telemetry(token: Token) -> None:
    _CALLS.reset(token)


def estimate_tokens(parts: Iterable[Any]) -> int:
    chars = sum(len(str(part or "")) for part in parts)
    return max(0, round(chars / 4.0))


def provider_usage(completion: Any) -> dict[str, int] | None:
    """Extract OpenAI-compatible usage metadata when a provider returns it."""
    usage = getattr(completion, "usage", None)
    if usage is None and isinstance(completion, dict):
        usage = completion.get("usage")
    if usage is None:
        return None

    def read(name: str) -> int:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    input_tokens = read("prompt_tokens") or read("input_tokens")
    output_tokens = read("completion_tokens") or read("output_tokens")
    total_tokens = read("total_tokens") or input_tokens + output_tokens
    if total_tokens <= 0 and input_tokens <= 0 and output_tokens <= 0:
        return None
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def record_llm_call(
    *,
    provider: str,
    model: str | None,
    operation: str,
    latency_ms: float,
    prompt_parts: Iterable[Any] = (),
    completion_text: str = "",
    usage: dict[str, int] | None = None,
    success: bool = True,
    error_type: str | None = None,
) -> dict[str, Any]:
    """Record counts and timing only; prompt/completion content is discarded."""
    actual = usage or {}
    actual_input = _int(actual.get("input_tokens"))
    actual_output = _int(actual.get("output_tokens"))
    actual_total = _int(actual.get("total_tokens")) or actual_input + actual_output
    usage_basis = "provider_reported" if actual_total > 0 else "chars_div_4_estimate"
    input_tokens = actual_input if usage_basis == "provider_reported" else estimate_tokens(prompt_parts)
    output_tokens = actual_output if usage_basis == "provider_reported" else estimate_tokens([completion_text])
    total_tokens = actual_total if usage_basis == "provider_reported" else input_tokens + output_tokens
    estimated_cost = _estimated_cost(input_tokens, output_tokens)
    item = {
        "provider": str(provider or "unknown"),
        "model": str(model or "unknown"),
        "operation": str(operation or "unspecified"),
        "success": bool(success),
        "error_type": str(error_type)[:80] if error_type else None,
        "latency_ms": round(max(0.0, float(latency_ms or 0.0)), 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "usage_basis": usage_basis,
        "estimated_cost_usd": estimated_cost,
    }
    calls = _CALLS.get()
    if calls is not None:
        calls.append(item)
    return item


def snapshot_llm_telemetry() -> dict[str, Any]:
    calls = list(_CALLS.get() or [])
    provider_calls = [item for item in calls if item["usage_basis"] == "provider_reported"]
    estimated_calls = [item for item in calls if item["usage_basis"] != "provider_reported"]
    total_input = sum(item["input_tokens"] for item in calls)
    total_output = sum(item["output_tokens"] for item in calls)
    total_tokens = sum(item["total_tokens"] for item in calls)
    total_cost = round(sum(float(item["estimated_cost_usd"]) for item in calls), 8)
    total_latency = round(sum(float(item["latency_ms"]) for item in calls), 2)
    return {
        "schema_version": "llm_usage_telemetry_v1",
        "call_count": len(calls),
        "successful_call_count": sum(1 for item in calls if item["success"]),
        "failed_call_count": sum(1 for item in calls if not item["success"]),
        "provider_reported_call_count": len(provider_calls),
        "estimated_call_count": len(estimated_calls),
        "actual_usage_coverage_rate": (
            round(len(provider_calls) / len(calls), 4) if calls else 0.0
        ),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_tokens,
        "estimated_cost_usd": total_cost,
        "llm_call_latency_ms": total_latency,
        "pricing_assumptions": {
            "input_cost_per_million_tokens_usd": _INPUT_COST_PER_MILLION_USD,
            "output_cost_per_million_tokens_usd": _OUTPUT_COST_PER_MILLION_USD,
            "source": "environment_or_engineering_default",
            "audited_billing": False,
        },
        "calls": calls,
        "content_retained": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _estimated_cost(input_tokens: int, output_tokens: int) -> float:
    return round(
        input_tokens / 1_000_000.0 * _INPUT_COST_PER_MILLION_USD
        + output_tokens / 1_000_000.0 * _OUTPUT_COST_PER_MILLION_USD,
        8,
    )


def _int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


__all__ = [
    "CLAIM_BOUNDARY",
    "LLMCallTimer",
    "estimate_tokens",
    "provider_usage",
    "record_llm_call",
    "reset_llm_telemetry",
    "snapshot_llm_telemetry",
    "start_llm_telemetry",
]
