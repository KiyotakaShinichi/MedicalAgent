"""Shaping a pipeline outcome into the response envelope callers receive.

Whatever branch ran, the result leaves through here: the answer, its citations
and sources, the safety decision, and the deterministic turn trace that makes a
reply auditable after the fact.

Two things in here are load-bearing rather than cosmetic:

* `_prefer_stricter_safety` merges two safety decisions by taking the stricter
  one. A merge that preferred the *later* decision would let a permissive
  post-generation result overwrite an earlier block.
* the trace records the model, usage, cost, refusal state and route reason. It
  is what a reviewer reads to understand why a given answer was produced, so
  losing a field makes past answers unexplainable rather than merely
  under-logged.
"""

from __future__ import annotations

from time import perf_counter
from types import SimpleNamespace

from backend.services.agent_cache import (
    _cache_policy_snapshot,
    _datetime_to_iso,
    store_cache,
)
from backend.services.agent_eval_log import _store_rag_evaluation_log
from backend.services.agent_eval_scoring import evaluate_rag_response
from backend.services.agent_output_gate import output_guardrail_check
from backend.services.agent_post_gen import (
    _apply_intent_aware_rag_layer,
    _apply_post_gen_validator,
)
from backend.services.rag_evidence_envelope import (
    EvidenceDisposition,
    enforce_evidence_release,
    enforce_transport_release,
)


def _prefer_stricter_safety(recomputed, precomputed):
    """Preserve contextual safety metadata without allowing a downgrade."""

    if not isinstance(precomputed, dict):
        return recomputed
    action_rank = {
        "ALLOW_EDUCATIONAL": 0,
        "ALLOW_WITH_BOUNDARY": 1,
        "SAFE_REDIRECT": 2,
        "REFUSE_ACTIONABLE": 3,
        "URGENT_ESCALATION": 4,
        "FAIL_CLOSED": 5,
    }
    recomputed_action_rank = action_rank.get(str(recomputed.get("policy_action")), -1)
    precomputed_action_rank = action_rank.get(str(precomputed.get("policy_action")), -1)
    if precomputed_action_rank < recomputed_action_rank:
        return recomputed
    if precomputed_action_rank > recomputed_action_rank:
        return precomputed
    level_rank = {
        "low_risk": 0,
        "moderate_risk": 1,
        "high_risk": 2,
        "blocked": 3,
    }
    recomputed_rank = level_rank.get(str(recomputed.get("level")), 0)
    precomputed_rank = level_rank.get(str(precomputed.get("level")), 0)
    if precomputed_rank < recomputed_rank:
        return recomputed
    if precomputed_rank > recomputed_rank:
        return precomputed
    if precomputed.get("context_reused") and not recomputed.get("context_reused"):
        return precomputed
    return recomputed


def _finalize_result(
    db,
    patient_id,
    query,
    rewritten,
    result,
    retrieved,
    reranked,
    compressed,
    input_guardrails,
    started,
    compound_intent=None,
    cache_write=None,
):
    """Orchestrate the post-generation pipeline:

      1. Run legacy output-guardrail heuristics.
      2. Run the post-gen safety validator (may rewrite the reply).
      3. Run the intent-aware RAG layer (mode -> tier filter -> claim
         validation -> evidence grade -> optional insufficient-evidence
         substitution).
      4. Compute end-to-end latency after all safety/governance work.
      5. Build the RAG evaluation telemetry block.
      6. Persist the RAGEvaluationLog row.

    Each step lives in a named helper so the failure surface is explicit
    and the call site reads top-to-bottom.
    """
    validation_errors = []
    post_generation_started = perf_counter()
    try:
        output_guardrails = output_guardrail_check(result)
    except Exception as exc:  # noqa: BLE001 - output validation must fail closed
        validation_errors.append(f"output_guardrail_exception:{type(exc).__name__}")
        output_guardrails = {"status": "failed", "issues": ["output_guardrail_exception"]}
    try:
        output_guardrails, pgv_decision = _apply_post_gen_validator(result, output_guardrails)
    except Exception as exc:  # noqa: BLE001 - safety validator unavailability denies release
        validation_errors.append(f"post_gen_validator_exception:{type(exc).__name__}")
        pgv_decision = SimpleNamespace(decision="unavailable")
        result["post_gen_validator"] = {
            "decision": "unavailable",
            "error_code": "post_gen_validator_exception",
            "exception_type": type(exc).__name__,
            "raw_response_logged": False,
        }
    post_generation_finished = perf_counter()
    try:
        _apply_intent_aware_rag_layer(result, retrieved, input_guardrails, pgv_decision)
    except Exception as exc:  # noqa: BLE001 - defense in depth around the layer itself
        validation_errors.append(f"rag_governance_exception:{type(exc).__name__}")
        result["rag_governance_error"] = {
            "status": "failed",
            "stage": "intent_aware_rag_layer",
            "code": "rag_governance_boundary_exception",
            "exception_type": type(exc).__name__,
            "raw_query_logged": False,
        }
        result["citations"] = []
    if isinstance(result.get("rag_governance_error"), dict):
        validation_errors.append(
            str(result["rag_governance_error"].get("code") or "rag_governance_failure")
        )
    governance_finished = perf_counter()
    pipeline_trace = result.setdefault("pipeline_trace", {})
    stage_ms = pipeline_trace.setdefault("stage_ms", {})
    stage_ms["post_generation_validation_ms"] = round(
        (post_generation_finished - post_generation_started) * 1000,
        2,
    )
    stage_ms["source_governance_ms"] = round(
        (governance_finished - post_generation_finished) * 1000,
        2,
    )
    latency_ms = round((governance_finished - started) * 1000, 2)
    from backend.services.llm_telemetry import snapshot_llm_telemetry

    try:
        llm_telemetry = snapshot_llm_telemetry()
    except Exception as exc:  # noqa: BLE001 - evidence answers require auditable completion
        llm_telemetry = {}
        validation_errors.append(f"telemetry_snapshot_exception:{type(exc).__name__}")
    if llm_telemetry.get("call_count"):
        result["llm_telemetry"] = llm_telemetry

    try:
        rag_evaluation = evaluate_rag_response(
            query=query,
            rewritten=rewritten,
            result=result,
            retrieved=retrieved,
            reranked=reranked,
            compressed=compressed,
            input_guardrails=input_guardrails,
            output_guardrails=output_guardrails,
            latency_ms=latency_ms,
        )
    except Exception as exc:  # noqa: BLE001 - do not translate evaluation failure into success
        validation_errors.append(f"rag_evaluation_exception:{type(exc).__name__}")
        rag_evaluation = {
            "status": "failed_closed",
            "reason": "rag_evaluation_exception",
            "clinical_validation": False,
        }
    result["guardrails"] = {
        "input":  input_guardrails,
        "output": output_guardrails,
    }
    result["rag_evaluation"] = rag_evaluation
    if compound_intent is not None:
        result["compound_intent"] = compound_intent
    enforce_evidence_release(
        result,
        query=query,
        retrieved=compressed or retrieved,
        input_guardrails=input_guardrails,
        validation_errors=validation_errors,
    )
    trace_ok = _attach_turn_trace(
        result=result,
        patient_id=patient_id,
        input_guardrails=input_guardrails,
        output_guardrails=output_guardrails,
        latency_ms=latency_ms,
    )
    evidence_required = bool((result.get("evidence_envelope") or {}).get("evidence_required"))
    if evidence_required and not trace_ok:
        validation_errors.append("trace_validation_or_persistence_failure")
        enforce_evidence_release(
            result,
            query=query,
            retrieved=compressed or retrieved,
            input_guardrails=input_guardrails,
            validation_errors=validation_errors,
        )
    try:
        _store_rag_evaluation_log(
            db=db,
            patient_id=patient_id,
            query=query,
            result=result,
            rag_evaluation=rag_evaluation,
            retrieved=retrieved,
            compressed=compressed,
        )
    except Exception as exc:  # noqa: BLE001 - logging failure cannot authorize evidence output
        try:
            db.rollback()
        except Exception:  # noqa: BLE001
            pass
        if evidence_required:
            validation_errors.append(f"rag_evaluation_log_exception:{type(exc).__name__}")
            enforce_evidence_release(
                result,
                query=query,
                retrieved=compressed or retrieved,
                input_guardrails=input_guardrails,
                validation_errors=validation_errors,
            )
        else:
            result.setdefault("evidence_envelope_events", []).append({
                "event": "non_evidence_observability_failure",
                "error_code": "rag_evaluation_log_exception",
                "raw_query_logged": False,
            })

    disposition = str((result.get("evidence_envelope") or {}).get("final_disposition") or "")
    if cache_write and disposition == EvidenceDisposition.ALLOW.value:
        try:
            cache_row = store_cache(
                db,
                cache_write["rewritten"],
                cache_write["intent"],
                cache_write["safety"],
                result,
                knowledge_fingerprint=cache_write["knowledge_fingerprint"],
            )
            result["cache"] = {
                "status": "stored",
                "cache_id": cache_row.id,
                "cacheable": True,
                "expires_at": _datetime_to_iso(cache_row.expires_at),
                "knowledge_fingerprint": cache_row.knowledge_fingerprint,
                "policy": _cache_policy_snapshot(cache_row.knowledge_fingerprint),
            }
        except Exception as exc:  # noqa: BLE001 - a cache write is not evidence authorization
            try:
                db.rollback()
            except Exception:  # noqa: BLE001
                pass
            result["cache"] = {
                "status": "not_stored_cache_error",
                "cacheable": False,
                "reason": f"cache_write_exception:{type(exc).__name__}",
            }
            result.setdefault("cache_rejection_events", []).append({
                "event": "rag_cache_rejected",
                "lookup": "write",
                "reasons": [f"cache_write_exception:{type(exc).__name__}"],
                "raw_query_logged": False,
            })
    elif cache_write:
        result["cache"] = {
            "status": "not_stored_release_denied",
            "cacheable": False,
            "reason": disposition or "missing_release_disposition",
        }
    return enforce_transport_release(result, query=query)


def _attach_turn_trace(result, patient_id, input_guardrails, output_guardrails, latency_ms):
    """Attach discrete diagnostics without private chain-of-thought."""
    try:
        from backend.services.agent_turn_trace import build_turn_trace, validate_trace_payload
        from backend.services.request_context import get_request_id
        from backend.services.trace_envelope_v2 import build_trace_envelope_v2, validate_trace_envelope_v2

        pipeline = result.get("pipeline_trace") or {}
        safety = result.get("safety") or {}
        llm_telemetry = result.get("llm_telemetry") or {}
        trace = build_turn_trace(
            correlation_id=get_request_id(),
            model_used={
                "answer": _trace_model_label(llm_telemetry),
                "route": pipeline.get("terminal_step"),
            },
            safety_scope={
                "level": safety.get("level") or input_guardrails.get("level"),
                "scope": safety.get("scope") or input_guardrails.get("scope"),
                "matched_terms": safety.get("matched_terms") or input_guardrails.get("matched_terms") or [],
            },
            intent={
                "deterministic_intent": result.get("intent"),
                "route_chosen": result.get("rag_mode") or pipeline.get("terminal_step"),
                "route_alternatives_considered": [
                    "deterministic_refusal",
                    "data_entry_confirmation",
                    "portal_help",
                    "source_governed_rag",
                    "insufficient_evidence",
                ],
                "why_route_was_chosen": _trace_route_reason(result),
            },
            retrieval_summary=result.get("retrieval_confidence") or {},
            emotional_distress=result.get("emotional_distress") or {},
            post_gen_validator=result.get("post_gen_validator") or {
                "output_guardrail_status": output_guardrails.get("status"),
                "output_issues": output_guardrails.get("issues") or [],
            },
            refusal={
                "refused": _trace_refused(result),
                "refusal_reason": _trace_refusal_reason(result),
            },
            cache=result.get("cache") or {},
            compound_intent=(
                result.get("compound_intent").to_dict()
                if hasattr(result.get("compound_intent"), "to_dict")
                else result.get("compound_intent") or {}
            ),
            latency_ms={
                "total": latency_ms,
                **((pipeline.get("stage_ms") or {}) if isinstance(pipeline, dict) else {}),
            },
            validator_latency_ms=((pipeline.get("stage_ms") or {}) if isinstance(pipeline, dict) else {}).get("post_generation_validation_ms"),
            patient_id=patient_id,
        ).to_dict()
        ok, problems = validate_trace_payload(trace)
        result["turn_trace"] = trace if ok else {"schema_version": "1.0", "validation_errors": problems}
        trace_v2 = build_trace_envelope_v2(
            result,
            patient_id=patient_id,
            route=str(result.get("rag_mode") or pipeline.get("terminal_step") or "patient_chat"),
            latency_ms={
                "total": latency_ms,
                **((pipeline.get("stage_ms") or {}) if isinstance(pipeline, dict) else {}),
            },
            correlation_id=get_request_id(),
            estimated_cost=_trace_usage_cost(llm_telemetry),
        )
        ok_v2, problems_v2 = validate_trace_envelope_v2(trace_v2)
        result["turn_trace_v2"] = (
            trace_v2
            if ok_v2
            else {
                "schema_version": "2.0",
                "validation_errors": problems_v2,
                "clinical_validation": False,
            }
        )
        return bool(ok and ok_v2)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never break chat
        result["turn_trace"] = {
            "schema_version": "1.0",
            "diagnostics_error": "trace_construction_failed",
            "exception_type": type(exc).__name__,
        }
        result["turn_trace_v2"] = {
            "schema_version": "2.0",
            "diagnostics_error": "trace_construction_failed",
            "exception_type": type(exc).__name__,
            "clinical_validation": False,
        }
        return False


def _trace_model_label(llm_telemetry):
    calls = llm_telemetry.get("calls") if isinstance(llm_telemetry, dict) else None
    if not isinstance(calls, list) or not calls:
        return "deterministic_local_or_untracked"
    labels = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        label = f"{call.get('provider')}/{call.get('model')}"
        if label not in labels:
            labels.append(label)
    return ",".join(labels)[:255] or "deterministic_local_or_untracked"


def _trace_usage_cost(llm_telemetry):
    if not isinstance(llm_telemetry, dict) or not llm_telemetry.get("call_count"):
        return {"available": False, "reason": "no_provider_call_captured"}
    return {
        "available": True,
        "estimated_cost_usd": llm_telemetry.get("estimated_cost_usd"),
        "input_tokens": llm_telemetry.get("input_tokens"),
        "output_tokens": llm_telemetry.get("output_tokens"),
        "total_tokens": llm_telemetry.get("total_tokens"),
        "usage_basis": (
            "provider_reported"
            if llm_telemetry.get("provider_reported_call_count")
            else "chars_div_4_estimate"
        ),
        "audited_billing": False,
    }


def _trace_refused(result):
    terminal = str((result.get("pipeline_trace") or {}).get("terminal_step") or "").lower()
    post = result.get("post_gen_validator") or {}
    evidence = result.get("evidence_grade") or {}
    return "refusal" in terminal or post.get("decision") == "blocked" or evidence.get("grade") == "insufficient"


def _trace_refusal_reason(result):
    if (result.get("post_gen_validator") or {}).get("decision") == "blocked":
        return "post_generation_validator_blocked"
    evidence = result.get("evidence_grade") or {}
    if evidence.get("grade") == "insufficient":
        return "insufficient_evidence"
    terminal = str((result.get("pipeline_trace") or {}).get("terminal_step") or "")
    if "refusal" in terminal:
        return terminal
    return None


def _trace_route_reason(result):
    safety = result.get("safety") or {}
    if safety.get("level") == "high_risk":
        return "high_risk_safety_scope"
    if result.get("retrieval_confidence"):
        return (result.get("retrieval_confidence") or {}).get("answerability_status")
    if result.get("cache"):
        return (result.get("cache") or {}).get("status")
    return (result.get("pipeline_trace") or {}).get("terminal_step")
