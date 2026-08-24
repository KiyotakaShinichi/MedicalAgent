"""Patient-agent pipeline execution: which branch runs, and in what order.

One request takes exactly one of four routes, and this module decides which:

* the **input guardrail** blocked it, so nothing downstream runs;
* a **cache hit** answers it without retrieval or generation;
* a **direct support** answer needs no retrieval;
* otherwise the full **retrieval and generation** path.

Ordering here is a safety property, not a performance detail. The safety scope
check runs before retrieval, so a blocked query never reaches the knowledge
base or the model, and the cache is consulted only after the guardrail has
passed - otherwise a cached answer could bypass a block that a later policy
change introduced.

Shaping the chosen branch's output into the response envelope is
`agent_result_shaping`; this module decides and executes, that one presents.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from time import perf_counter

from backend.services.agent_answer_composition import (
    _uses_direct_support_lane,
    generate_answer,
    validate_answer_and_citations,
)
from backend.services.agent_cache import (
    _cache_policy_snapshot,
    _cache_rejection_reason,
    _lookup_cache,
    is_cacheable,
)
from backend.services.agent_input_gate import (
    _security_block_reply,
    input_guardrail_check,
)
from backend.services.agent_intent_router import (
    _validated_preselected_intent,
    route_intent,
)
from backend.services.agent_kb_corpus import knowledge_base_fingerprint
from backend.services.agent_query_rewriting import rewrite_and_decompose
from backend.services.agent_result_shaping import (
    _finalize_result,
    _prefer_stricter_safety,
)
from backend.services.agent_retrieval import (
    contextual_compression,
    expand_parent_child_windows,
    hybrid_retrieval,
    rerank_context,
)
from backend.services.agent_safety import safety_scope_check
from backend.services.agent_trace import _trace


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
    if not isinstance(patient_context, Mapping):
        raise ValueError("malformed_patient_context")
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
