"""RAG-evaluation log writer for the patient agent.

Persists one ``RAGEvaluationLog`` row per chat turn.  Captures
intent + safety + cache state + the four engineering proxies from
:func:`agent_eval_scoring.evaluate_rag_response` + the retrieval and
citation source ids + the intent-aware RAG layer's tier-filter /
claim-validation / evidence-grade envelopes.

The row is what the admin trace-replay panel reads back.  Failing to
write it must not crash the pipeline, but right now the writer is
called inside ``_finalize_result`` which already runs inside a try
block, so we keep the implementation simple here.

Extracted from ``agent_rag.py`` as part of the agent_rag.py module
split.  Re-exported from ``backend.services.agent_rag`` so the
internal call site in ``_finalize_result`` keeps working.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from backend.models import RAGEvaluationLog
from backend.services.agent_cache import _query_hash
from backend.services.agent_query_rewriting import _normalize_query
from backend.services.pii_redaction import redact_text
from backend.services.request_context import get_request_id


def store_rag_evaluation_log(
    db,
    patient_id: str,
    query: str,
    result: Mapping[str, Any],
    rag_evaluation: Mapping[str, Any],
    retrieved: list[Mapping[str, Any]],
    compressed: list[Mapping[str, Any]],  # noqa: ARG001 — kept for signature parity with the original call site
) -> Any:
    """Insert one ``RAGEvaluationLog`` row from the evaluated turn.

    ``compressed`` is accepted (and ignored) for signature parity with
    the original inline function so existing call sites pass through
    unchanged.
    """
    hallucination       = rag_evaluation["hallucination"]
    grounding           = rag_evaluation["answer_grounding"]
    retrieval_precision = rag_evaluation["retrieval_precision_at_3"]
    cost_latency        = rag_evaluation["cost_latency"]
    guardrails          = rag_evaluation["guardrail_summary"]

    row = RAGEvaluationLog(
        patient_id=patient_id,
        request_id=get_request_id(),
        query_hash=_query_hash(_normalize_query(query)),
        query_preview=redact_text(str(query or ""))[:120],
        intent=result.get("intent") or "unknown",
        safety_level=(result.get("safety") or {}).get("level") or "unknown",
        cache_status=(result.get("cache") or {}).get("status"),
        terminal_step=(result.get("pipeline_trace") or {}).get("terminal_step"),
        retrieval_precision_at_3=retrieval_precision.get("value"),
        grounding_score=grounding.get("score"),
        hallucination_score=hallucination.get("score"),
        hallucination_risk=hallucination.get("risk"),
        input_guardrail_status=guardrails.get("input_status"),
        output_guardrail_status=guardrails.get("output_status"),
        latency_ms=cost_latency.get("latency_ms"),
        estimated_input_tokens=cost_latency.get("estimated_input_tokens"),
        estimated_output_tokens=cost_latency.get("estimated_output_tokens"),
        estimated_total_tokens=cost_latency.get("estimated_total_tokens"),
        estimated_llm_cost_usd=cost_latency.get("estimated_llm_cost_usd"),
        stage_latency_json=json.dumps(cost_latency.get("stage_ms") or {}),
        token_usage_json=json.dumps(cost_latency.get("provider_token_usage") or {}),
        model_used=_model_used(result),
        retrieved_source_ids_json=json.dumps([item.get("id") for item in retrieved if item.get("id")]),
        cited_source_ids_json=json.dumps([item.get("id") for item in result.get("citations") or []]),
        guardrail_issues_json=json.dumps({
            "input":  guardrails.get("input_issues") or [],
            "output": guardrails.get("output_issues") or [],
        }),
        rag_mode=result.get("rag_mode"),
        rewritten_query=result.get("rewritten_query"),
        evidence_grade_json=     json.dumps(result.get("evidence_grade"))      if result.get("evidence_grade")      is not None else None,
        claim_validation_json=   json.dumps(result.get("claim_validation"))    if result.get("claim_validation")    is not None else None,
        tier_filter_json=        json.dumps(result.get("tier_filter"))         if result.get("tier_filter")         is not None else None,
        post_gen_validator_json= json.dumps(result.get("post_gen_validator"))  if result.get("post_gen_validator")  is not None else None,
        compound_intent_json=    json.dumps(_compound_intent_log_payload(result.get("compound_intent"))) if result.get("compound_intent") is not None else None,
        retrieval_confidence_json=json.dumps(result.get("retrieval_confidence")) if result.get("retrieval_confidence") is not None else None,
        trace_diagnostics_json=  json.dumps(result.get("turn_trace")) if result.get("turn_trace") is not None else None,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _compound_intent_log_payload(value):
    """Normalize whatever the chat layer stuffed into ``result["compound_intent"]``.

    The chat layer passes a ``CompoundIntent`` dataclass (which has a
    ``.to_dict()`` method) when called from support_chat_agent, or a
    plain dict when callers passed one in directly.  We tolerate both
    + None.  Falls back to ``str(value)`` only if everything else fails."""
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, dict):
        return value
    return {"raw": str(value)[:240]}


def _model_used(result: Mapping[str, Any]) -> str:
    """Best-effort model/provider label for cost reporting.

    Default local runs are deterministic/template-driven, so there may be no
    provider metadata. Keep that explicit instead of pretending an API model
    was called.
    """
    telemetry = result.get("llm_telemetry") or {}
    calls = telemetry.get("calls") if isinstance(telemetry, Mapping) else None
    if isinstance(calls, list) and calls:
        labels = []
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            provider = call.get("provider")
            model = call.get("model")
            label = f"{provider}/{model}" if provider and model else provider or model
            if label and label not in labels:
                labels.append(str(label))
        if labels:
            return ",".join(labels)[:255]
    llm = result.get("llm") or result.get("llm_metadata") or {}
    if isinstance(llm, Mapping):
        provider = llm.get("provider")
        model = llm.get("model")
        if provider and model:
            return f"{provider}/{model}"
        if provider:
            return str(provider)
    if (result.get("pipeline_trace") or {}).get("terminal_step") in {"safety_refusal", "security_refusal"}:
        return "deterministic_refusal"
    return "deterministic_local_or_untracked"


# Back-compat alias — agent_rag._finalize_result calls the underscore name.
_store_rag_evaluation_log = store_rag_evaluation_log


__all__ = ["store_rag_evaluation_log", "_store_rag_evaluation_log"]
