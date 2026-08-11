"""Patient-agent orchestrator (the residue after the 14-slice carve-out).

This module is now just the entry point + post-generation orchestrator.
Every other concern lives in a dedicated module:

  safety scope            -> agent_safety
  input / output gates    -> agent_input_gate / agent_output_gate
  intent routing          -> agent_intent_router
  query rewriting         -> agent_query_rewriting
  retrieval + rerank      -> agent_retrieval
  answer composition      -> agent_answer_composition
  post-gen + RAG layer    -> agent_post_gen
  cache                   -> agent_cache
  trace envelope          -> agent_trace
  eval scoring            -> agent_eval_scoring
  RAG eval log writer     -> agent_eval_log
  KB seed corpus          -> agent_knowledge_snippets
  KB merge cache          -> agent_kb_corpus

The block below re-imports every public symbol (and every underscore
alias the in-module orchestrator used inline before the split) so
external callers can keep ``from backend.services.agent_rag import X``
working unchanged.  External call sites: chat, eval scripts, tests,
admin scripts — six places at last count.
"""
from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from types import SimpleNamespace

# ─── Consolidated re-export shim ─────────────────────────────────────────────
#
# All symbols below were previously defined inline in this file.  Each
# extracted module preserves the public name + (where applicable) an
# underscore alias.  Keeping the re-exports in one block (instead of 14
# scattered ``from ... import`` blocks) makes the residual orchestrator
# code readable top-to-bottom.

from backend.services.agent_answer_composition import (  # noqa: F401
    REFUSAL_INTENTS,
    _contains_diagnostic_or_treatment_claim,
    _safety_reply,
    _uses_direct_support_lane,
    generate_answer,
    validate_answer_and_citations,
)
from backend.services.agent_cache import (  # noqa: F401
    AGENT_CACHE_SCHEMA_VERSION,
    AGENT_CACHE_TTL_DAYS,
    SEMANTIC_CACHE_MIN_SIMILARITY,
    _cache_policy_snapshot,
    _cache_rejection_reason,
    _cache_response_payload,
    _cache_row_freshness,
    _cache_row_policy,
    _coerce_utc,
    _datetime_to_iso,
    _json_loads,
    _mark_cache_hit,
    _query_hash,
    exact_cache_check,
    is_cacheable,
    semantic_cache_check,
    store_cache,
)
from backend.services.agent_eval_log import _store_rag_evaluation_log  # noqa: F401
from backend.services.agent_eval_scoring import (  # noqa: F401
    _content_tokens,
    _cost_latency_note,
    _estimate_tokens,
    _maybe_run_llm_judge,
    _score_status,
    answer_grounding_score,
    estimate_token_and_cost,
    evaluate_rag_response,
    hallucination_score,
    proxy_retrieval_precision_at_k,
)
from backend.services.agent_input_gate import (  # noqa: F401
    _security_block_reply,
    input_guardrail_check,
)
from backend.services.agent_intent_router import (  # noqa: F401
    _is_conversation_opening,
    _is_identity_or_capability_question,
    _is_social_checkin,
    route_intent,
)
from backend.services.agent_kb_corpus import (  # noqa: F401
    _invalidate_kb_cache,
    _knowledge_snippets,
    get_rag_corpus,
    knowledge_base_fingerprint,
)
from backend.services.agent_knowledge_snippets import KNOWLEDGE_SNIPPETS  # noqa: F401
from backend.services.agent_output_gate import output_guardrail_check  # noqa: F401
from backend.services.agent_post_gen import (  # noqa: F401
    _apply_intent_aware_rag_layer,
    _apply_post_gen_validator,
)
from backend.services.agent_query_rewriting import (  # noqa: F401
    _normalize_query,
    _semantic_key,
    _tokenize,
    rewrite_and_decompose,
)
from backend.services.agent_retrieval import (  # noqa: F401
    CURATED_SOURCES as _CURATED_SOURCES,
    MAX_CONTEXT_CHARS,
    _cross_encoder_enabled,
    _cross_encoder_scores,
    _domain_boost,
    _get_cross_encoder,
    _intent_boost,
    _reranker_backend,
    _section_boost,
    contextual_compression,
    expand_parent_child_windows,
    hybrid_retrieval,
    rerank_context,
)
from backend.services.agent_safety import safety_scope_check  # noqa: F401
from backend.services.agent_trace import _trace  # noqa: F401
from backend.services.rag_evidence_envelope import (
    EvidenceDisposition,
    build_fail_closed_error_result,
    enforce_evidence_release,
    enforce_transport_release,
)

# ``route_intent_with_local_llm`` is consulted by ``agent_intent_router.route_intent``
# via an attribute lookup on THIS module (``agent_rag``).  The indirection
# preserves the long-standing test contract: monkey-patching
# ``agent_rag.route_intent_with_local_llm`` must be authoritative for the
# intent router, even after the function lives in ``local_llm``.
from backend.services.local_llm import route_intent_with_local_llm  # noqa: F401


# ─── Orchestrator entry point ────────────────────────────────────────────────


def run_patient_agent_pipeline(
    db,
    patient_id,
    query,
    patient_context,
    fallback_response,
    actions=None,
    urgent_flags=None,
    preselected_intent=None,
    compound_intent=None,
    precomputed_safety=None,
):
    """Run the patient agent with a final deny-on-exception boundary."""
    try:
        return _run_patient_agent_pipeline_impl(
            db=db,
            patient_id=patient_id,
            query=query,
            patient_context=patient_context,
            fallback_response=fallback_response,
            actions=actions,
            urgent_flags=urgent_flags,
            preselected_intent=preselected_intent,
            compound_intent=compound_intent,
            precomputed_safety=precomputed_safety,
        )
    except Exception as exc:  # noqa: BLE001 - no pipeline exception may leak a candidate
        return build_fail_closed_error_result(
            query=query,
            error_code=f"patient_agent_pipeline_exception:{type(exc).__name__}",
        )


def _run_patient_agent_pipeline_impl(
    db,
    patient_id,
    query,
    patient_context,
    fallback_response,
    actions=None,
    urgent_flags=None,
    preselected_intent=None,
    compound_intent=None,
    precomputed_safety=None,
):
    """Main agent entry point — a dispatcher.

    Resolves the safety envelope, runs the input guardrail, and routes
    to one of four branch handlers based on the outcome:

      1. ``_run_input_guardrail_block_branch``   — security / privacy fail
      2. ``_run_cache_hit_branch``               — exact or semantic cache hit
      3. ``_run_direct_support_branch``          — conversation / memory / timeline
      4. ``_run_rag_generation_branch``          — full retrieval + generation

    Every branch finalizes through ``_finalize_result``.
    """
    started = perf_counter()
    actions = actions or []
    urgent_flags = urgent_flags or []
    safety = _prefer_stricter_safety(
        safety_scope_check(query, urgent_flags),
        precomputed_safety,
    )
    input_guardrails = input_guardrail_check(query, safety)
    t_safety = perf_counter()

    if input_guardrails["status"] == "failed":
        return _run_input_guardrail_block_branch(
            db=db,
            patient_id=patient_id,
            query=query,
            safety=safety,
            input_guardrails=input_guardrails,
            started=started,
            compound_intent=compound_intent,
        )

    intent = _validated_preselected_intent(preselected_intent, safety) or route_intent(query, actions, safety)
    rewritten = rewrite_and_decompose(query, intent)
    t_routing = perf_counter()
    cacheable = is_cacheable(query, intent, safety, actions, urgent_flags)
    knowledge_fingerprint = knowledge_base_fingerprint()
    cache_policy = _cache_policy_snapshot(knowledge_fingerprint)

    cache_rejection_events = []
    cache_hit = _lookup_cache(
        db,
        cacheable,
        rewritten,
        intent,
        safety,
        knowledge_fingerprint,
        cache_rejection_events=cache_rejection_events,
    )
    if cache_hit:
        return _run_cache_hit_branch(
            db=db,
            patient_id=patient_id,
            query=query,
            rewritten=rewritten,
            intent=intent,
            safety=safety,
            cache_hit=cache_hit,
            cache_policy=cache_policy,
            input_guardrails=input_guardrails,
            started=started,
            cache_rejection_events=cache_rejection_events,
            compound_intent=compound_intent,
        )

    if _uses_direct_support_lane(intent, safety):
        return _run_direct_support_branch(
            db=db,
            patient_id=patient_id,
            query=query,
            rewritten=rewritten,
            intent=intent,
            safety=safety,
            fallback_response=fallback_response,
            actions=actions,
            patient_context=patient_context,
            cache_policy=cache_policy,
            input_guardrails=input_guardrails,
            started=started,
            cache_rejection_events=cache_rejection_events,
            compound_intent=compound_intent,
        )

    return _run_rag_generation_branch(
        db=db,
        patient_id=patient_id,
        query=query,
        rewritten=rewritten,
        intent=intent,
        safety=safety,
        fallback_response=fallback_response,
        actions=actions,
        urgent_flags=urgent_flags,
        patient_context=patient_context,
        cacheable=cacheable,
        knowledge_fingerprint=knowledge_fingerprint,
        cache_policy=cache_policy,
        input_guardrails=input_guardrails,
        started=started,
        t_safety=t_safety,
        t_routing=t_routing,
        cache_rejection_events=cache_rejection_events,
        compound_intent=compound_intent,
    )


def _prefer_stricter_safety(recomputed, precomputed):
    """Preserve contextual safety metadata without allowing a downgrade."""

    if not isinstance(precomputed, dict):
        return recomputed
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


# ─── Branch handlers ─────────────────────────────────────────────────────────


def _run_input_guardrail_block_branch(
    *, db, patient_id, query, safety, input_guardrails, started,
    compound_intent=None,
):
    """Branch 1: input guardrail rejected the request entirely.  Returns
    the deterministic security refusal — no retrieval, no generation."""
    safety = {
        **safety,
        "level": "high_risk",
        "scope": input_guardrails["scope"],
        "cache_allowed": False,
        "message": input_guardrails["message"],
    }
    intent = "security_boundary"
    rewritten = rewrite_and_decompose(query, intent)
    result = {
        "reply": _security_block_reply(input_guardrails),
        "citations": [],
        "intent": intent,
        "safety": safety,
        "retrieval_context": [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "safety_note": (
            "Security boundary: the assistant cannot reveal private records, system instructions, "
            "database contents, secrets, or raw internal knowledge base data."
        ),
        "validation": {
            "status": "passed",
            "issues": [],
            "citation_count": 0,
        },
        "cache": {
            "status": "blocked_by_input_guardrail",
            "cacheable": False,
            "reason": input_guardrails["scope"],
        },
        "pipeline_trace": _trace(safety, intent, rewritten, [], [], [], "input_guardrail_block"),
    }
    return _finalize_result(
        db=db,
        patient_id=patient_id,
        query=query,
        rewritten=rewritten,
        result=result,
        retrieved=[],
        reranked=[],
        compressed=[],
        input_guardrails=input_guardrails,
        started=started,
        compound_intent=compound_intent,
    )


def _lookup_cache(
    db,
    cacheable,
    rewritten,
    intent,
    safety,
    knowledge_fingerprint,
    cache_rejection_events=None,
):
    """Return the exact-cache hit envelope, falling back to semantic.
    None when the request isn't cacheable or no fresh row matches."""
    if not cacheable:
        return None
    hit = exact_cache_check(
        db,
        rewritten["normalized_query"],
        intent=intent,
        safety_level=safety.get("level"),
        knowledge_fingerprint=knowledge_fingerprint,
        rejection_events=cache_rejection_events,
    )
    if hit is None:
        hit = semantic_cache_check(
            db,
            rewritten["semantic_key"],
            intent,
            knowledge_fingerprint=knowledge_fingerprint,
            rejection_events=cache_rejection_events,
        )
    return hit


def _run_cache_hit_branch(
    *, db, patient_id, query, rewritten, intent, safety,
    cache_hit, cache_policy, input_guardrails, started,
    cache_rejection_events=None,
    compound_intent=None,
):
    """Branch 2: exact or semantic cache hit.  Reuses the stored
    response envelope and re-runs only the post-gen pipeline."""
    result = {
        **cache_hit["response"],
        "cache": {
            "status": cache_hit["status"],
            "cache_id": cache_hit["cache_id"],
            "cacheable": True,
            "expires_at": cache_hit.get("expires_at"),
            "knowledge_fingerprint": cache_hit.get("knowledge_fingerprint"),
            "policy": cache_hit.get("policy"),
        },
        "pipeline_trace": _trace(safety, intent, rewritten, [], [], [], "cache_hit", cache_policy=cache_policy),
        "cache_rejection_events": list(cache_rejection_events or []),
    }
    return _finalize_result(
        db=db,
        patient_id=patient_id,
        query=query,
        rewritten=rewritten,
        result=result,
        retrieved=[],
        reranked=[],
        compressed=result.get("retrieval_context") or [],
        input_guardrails=input_guardrails,
        started=started,
        compound_intent=compound_intent,
    )


def _run_direct_support_branch(
    *, db, patient_id, query, rewritten, intent, safety,
    fallback_response, actions, patient_context,
    cache_policy, input_guardrails, started,
    cache_rejection_events=None,
    compound_intent=None,
):
    """Branch 3: direct-support lane — return ``fallback_response``
    verbatim (conversation / memory / timeline / emotional / general).
    No retrieval, no caching."""
    generated = generate_answer(
        query=query,
        fallback_response=fallback_response,
        safety=safety,
        intent=intent,
        compressed_context=[],
        actions=actions,
        patient_context=patient_context,
    )
    validated = validate_answer_and_citations(generated, [], safety)
    result = {
        **validated,
        "cache": {
            "status": "not_cacheable",
            "cacheable": False,
            "reason": f"intent_not_cacheable:{intent}",
            "policy": cache_policy,
        },
        "pipeline_trace": _trace(safety, intent, rewritten, [], [], [], "direct_support", cache_policy=cache_policy),
        "cache_rejection_events": list(cache_rejection_events or []),
    }
    return _finalize_result(
        db=db,
        patient_id=patient_id,
        query=query,
        rewritten=rewritten,
        result=result,
        retrieved=[],
        reranked=[],
        compressed=[],
        input_guardrails=input_guardrails,
        started=started,
        compound_intent=compound_intent,
    )


def _run_rag_generation_branch(
    *, db, patient_id, query, rewritten, intent, safety,
    fallback_response, actions, urgent_flags, patient_context,
    cacheable, knowledge_fingerprint, cache_policy,
    input_guardrails, started, t_safety, t_routing,
    cache_rejection_events=None,
    compound_intent=None,
):
    """Branch 4: full RAG path — retrieve, rerank, compress, generate,
    validate.  Stores the response in the cache when validation passes
    and the request is cacheable.  Embeds per-stage latency in the
    pipeline_trace's ``stage_ms`` block."""
    from backend.services.rag_execution_policy import govern_candidates, plan_rag_execution

    t_retrieval_started = perf_counter()
    actor_role = (input_guardrails or {}).get("actor_role")
    execution_policy, mode = plan_rag_execution(intent=intent, rewritten=rewritten, actor_role=actor_role)
    retrieved = hybrid_retrieval(rewritten, intent)
    t_candidates = perf_counter()
    governed_retrieved, initial_filter_trace = govern_candidates(
        retrieved,
        mode,
        limit=execution_policy.max_governed_candidates or None,
    )
    expanded = (
        expand_parent_child_windows(governed_retrieved)
        if execution_policy.apply_parent_child
        else governed_retrieved
    )
    governed_expanded, expanded_filter_trace = govern_candidates(
        expanded,
        mode,
        limit=execution_policy.max_governed_candidates or None,
    )
    t_retrieval = perf_counter()
    if execution_policy.apply_reranker:
        reranked = rerank_context(governed_expanded, rewritten, intent, safety)
        reranker_telemetry = (
            (reranked[0].get("rerank_telemetry") or {})
            if reranked and isinstance(reranked[0], dict)
            else {}
        )
        if reranker_telemetry.get("enabled") is True and reranker_telemetry.get("available") is False:
            raise RuntimeError("configured_reranker_unavailable")
    else:
        reranked = [
            {
                **item,
                "rerank_score": item.get("retrieval_score", 0),
                "reranker_backend": "intent_policy_stage_skipped",
            }
            for item in governed_expanded
        ]
    t_rerank_only = perf_counter()
    compressed = contextual_compression(reranked)
    t_rerank = perf_counter()
    from backend.services.research_evidence_answerability import (
        assess_research_evidence_answerability,
    )

    research_answerability = assess_research_evidence_answerability(
        query=query,
        chunks=compressed,
        intent=intent,
        safety=safety,
    )
    if research_answerability.requires_abstention:
        generated = {
            "reply": research_answerability.safe_reply,
            "citations": [],
            "deliberate_evidence_abstention": True,
            "intent": intent,
            "safety": safety,
            "retrieval_context": compressed,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "safety_note": (
                "Engineering evidence abstention: related retrieval is not treated as claim support."
            ),
        }
    else:
        generated = generate_answer(
            query=query,
            fallback_response=fallback_response,
            safety=safety,
            intent=intent,
            compressed_context=compressed,
            actions=actions,
            patient_context=patient_context,
        )
    validated = validate_answer_and_citations(generated, compressed, safety)
    if not isinstance(validated, dict) or not isinstance(validated.get("validation"), dict):
        raise ValueError("malformed_generation_or_validation_output")
    if not str(validated.get("reply") or "").strip():
        raise ValueError("empty_or_truncated_generation_output")
    validated["research_evidence_answerability"] = research_answerability.to_dict()
    t_generation = perf_counter()

    cache_write = None
    if cacheable and validated["validation"].get("status") == "passed":
        cache_write = {
            "rewritten": rewritten,
            "intent": intent,
            "safety": safety,
            "knowledge_fingerprint": knowledge_fingerprint,
        }
        cache_status = {
            "status": "pending_release_authorization",
            "cacheable": True,
            "knowledge_fingerprint": knowledge_fingerprint,
            "policy": cache_policy,
        }
    else:
        cache_status = {
            "status": "not_cacheable",
            "cacheable": False,
            "reason": _cache_rejection_reason(query, intent, safety, actions, urgent_flags),
            "policy": cache_policy,
        }

    stage_ms = {
        "safety_gate_ms":     round((t_safety - started) * 1000, 2),
        "intent_routing_ms":  round((t_routing - t_safety) * 1000, 2),
        "retrieval_ms":       round((t_candidates - t_retrieval_started) * 1000, 2),
        "pre_generation_governance_ms": round((t_retrieval - t_candidates) * 1000, 2),
        "rerank_ms":          round((t_rerank_only - t_retrieval) * 1000, 2),
        "compression_ms":     round((t_rerank - t_rerank_only) * 1000, 2),
        "generation_ms":      round((t_generation - t_rerank) * 1000, 2),
    }
    result = {
        **validated,
        "cache": cache_status,
        "rag_execution_policy": execution_policy.to_dict(),
        "pregen_tier_filter": {
            "initial_retrieval": initial_filter_trace,
            "after_parent_child": expanded_filter_trace,
        },
        "cache_rejection_events": list(cache_rejection_events or []),
        "pipeline_trace": {
            **_trace(safety, intent, rewritten, retrieved, reranked, compressed, "generated", cache_policy=cache_policy),
            "stage_ms": stage_ms,
            "rag_execution_policy": execution_policy.to_dict(),
            "pregen_tier_filter": {
                "initial_retrieval": initial_filter_trace,
                "after_parent_child": expanded_filter_trace,
            },
        },
    }
    return _finalize_result(
        db=db,
        patient_id=patient_id,
        query=query,
        rewritten=rewritten,
        result=result,
        retrieved=retrieved,
        reranked=reranked,
        compressed=compressed,
        input_guardrails=input_guardrails,
        started=started,
        compound_intent=compound_intent,
        cache_write=cache_write,
    )


def _validated_preselected_intent(intent, safety):
    """Validate a caller-provided intent and downgrade to a safety
    boundary intent when the safety scope demands it."""
    allowed = {
        "safety_boundary",
        "treatment_decision_boundary",
        "data_entry_confirmation",
        "portal_help",
        "patient_timeline_monitoring",
        "education",
        "emotional_support",
        "general_support",
        "conversation",
        "patient_memory",
    }
    if intent not in allowed:
        return None
    if safety.get("scope") == "treatment_decision_request":
        return "treatment_decision_boundary"
    if safety.get("scope") in {"urgent_or_safety_related", "diagnosis_or_outcome_claim"}:
        return "safety_boundary"
    return intent


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
