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
    safety = safety_scope_check(query, urgent_flags)
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
        )

    intent = _validated_preselected_intent(preselected_intent, safety) or route_intent(query, actions, safety)
    rewritten = rewrite_and_decompose(query, intent)
    t_routing = perf_counter()
    cacheable = is_cacheable(query, intent, safety, actions, urgent_flags)
    knowledge_fingerprint = knowledge_base_fingerprint()
    cache_policy = _cache_policy_snapshot(knowledge_fingerprint)

    cache_hit = _lookup_cache(db, cacheable, rewritten, intent, safety, knowledge_fingerprint)
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
    )


# ─── Branch handlers ─────────────────────────────────────────────────────────


def _run_input_guardrail_block_branch(
    *, db, patient_id, query, safety, input_guardrails, started,
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
    )


def _lookup_cache(db, cacheable, rewritten, intent, safety, knowledge_fingerprint):
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
    )
    if hit is None:
        hit = semantic_cache_check(
            db, rewritten["semantic_key"], intent, knowledge_fingerprint=knowledge_fingerprint,
        )
    return hit


def _run_cache_hit_branch(
    *, db, patient_id, query, rewritten, intent, safety,
    cache_hit, cache_policy, input_guardrails, started,
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
    )


def _run_direct_support_branch(
    *, db, patient_id, query, rewritten, intent, safety,
    fallback_response, actions, patient_context,
    cache_policy, input_guardrails, started,
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
    )


def _run_rag_generation_branch(
    *, db, patient_id, query, rewritten, intent, safety,
    fallback_response, actions, urgent_flags, patient_context,
    cacheable, knowledge_fingerprint, cache_policy,
    input_guardrails, started, t_safety, t_routing,
):
    """Branch 4: full RAG path — retrieve, rerank, compress, generate,
    validate.  Stores the response in the cache when validation passes
    and the request is cacheable.  Embeds per-stage latency in the
    pipeline_trace's ``stage_ms`` block."""
    retrieved = hybrid_retrieval(rewritten, intent)
    expanded = expand_parent_child_windows(retrieved)
    t_retrieval = perf_counter()
    reranked = rerank_context(expanded, rewritten, intent, safety)
    compressed = contextual_compression(reranked)
    t_rerank = perf_counter()
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
    t_generation = perf_counter()

    if cacheable and validated["validation"]["status"] == "passed":
        cache_row = store_cache(db, rewritten, intent, safety, validated, knowledge_fingerprint=knowledge_fingerprint)
        cache_status = {
            "status": "stored",
            "cache_id": cache_row.id,
            "cacheable": True,
            "expires_at": _datetime_to_iso(cache_row.expires_at),
            "knowledge_fingerprint": cache_row.knowledge_fingerprint,
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
        "retrieval_ms":       round((t_retrieval - t_routing) * 1000, 2),
        "rerank_ms":          round((t_rerank - t_retrieval) * 1000, 2),
        "generation_ms":      round((t_generation - t_rerank) * 1000, 2),
    }
    result = {
        **validated,
        "cache": cache_status,
        "pipeline_trace": {
            **_trace(safety, intent, rewritten, retrieved, reranked, compressed, "generated", cache_policy=cache_policy),
            "stage_ms": stage_ms,
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
):
    """Orchestrate the post-generation pipeline:

      1. Compute latency.
      2. Run legacy output-guardrail heuristics.
      3. Run the post-gen safety validator (may rewrite the reply).
      4. Run the intent-aware RAG layer (mode -> tier filter -> claim
         validation -> evidence grade -> optional insufficient-evidence
         substitution).
      5. Build the RAG evaluation telemetry block.
      6. Persist the RAGEvaluationLog row.

    Each step lives in a named helper so the failure surface is explicit
    and the call site reads top-to-bottom.
    """
    latency_ms = round((perf_counter() - started) * 1000, 2)
    output_guardrails = output_guardrail_check(result)
    output_guardrails, pgv_decision = _apply_post_gen_validator(result, output_guardrails)
    _apply_intent_aware_rag_layer(result, retrieved, input_guardrails, pgv_decision)

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
    result["guardrails"] = {
        "input":  input_guardrails,
        "output": output_guardrails,
    }
    result["rag_evaluation"] = rag_evaluation
    _store_rag_evaluation_log(
        db=db,
        patient_id=patient_id,
        query=query,
        result=result,
        rag_evaluation=rag_evaluation,
        retrieved=retrieved,
        compressed=compressed,
    )
    return result
