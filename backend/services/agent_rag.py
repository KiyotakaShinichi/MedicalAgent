import hashlib
import importlib.util
import json
import os
import re
from time import perf_counter
from datetime import datetime, timezone, timedelta

from backend.models import AgentResponseCache, RAGEvaluationLog
from backend.services.kb_ingestion import load_ingested_chunks
from backend.services.local_llm import configured_llm_providers, decide_cache_with_local_llm, route_intent_with_local_llm
from backend.services.rag_vector_index import corpus_fingerprint, search_hybrid_index
from backend.services.pii_redaction import redact_text
from backend.services.request_context import get_request_id
from backend.services.security_guardrails import detect_multilingual_medical_danger, detect_prompt_injection_or_exfiltration, normalize_security_text


# MAX_CONTEXT_CHARS + _CROSS_ENCODER_CACHE moved to
# backend.services.agent_retrieval and are re-imported lower in this
# module so existing references still resolve.
# AGENT_CACHE_* + SEMANTIC_CACHE_MIN_SIMILARITY moved to
# backend.services.agent_cache and are re-imported lower in this module
# alongside the rest of the cache layer.

# _KB_CORPUS_CACHE moved to backend.services.agent_kb_corpus alongside
# the loader functions that own it.


# KNOWLEDGE_SNIPPETS moved to backend.services.agent_knowledge_snippets as
# part of the agent_rag.py module split (~290 lines of pure data).
from backend.services.agent_knowledge_snippets import KNOWLEDGE_SNIPPETS  # noqa: F401


def run_patient_agent_pipeline(db, patient_id, query, patient_context, fallback_response, actions=None, urgent_flags=None, preselected_intent=None):
    started = perf_counter()
    actions = actions or []
    urgent_flags = urgent_flags or []
    safety = safety_scope_check(query, urgent_flags)
    input_guardrails = input_guardrail_check(query, safety)
    t_safety = perf_counter()
    if input_guardrails["status"] == "failed":
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
            "safety_note": "Security boundary: the assistant cannot reveal private records, system instructions, database contents, secrets, or raw internal knowledge base data.",
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
    intent = _validated_preselected_intent(preselected_intent, safety) or route_intent(query, actions, safety)
    rewritten = rewrite_and_decompose(query, intent)
    t_routing = perf_counter()
    cacheable = is_cacheable(query, intent, safety, actions, urgent_flags)
    knowledge_fingerprint = knowledge_base_fingerprint()
    cache_policy = _cache_policy_snapshot(knowledge_fingerprint)

    cache_hit = None
    if cacheable:
        cache_hit = exact_cache_check(
            db,
            rewritten["normalized_query"],
            intent=intent,
            safety_level=safety.get("level"),
            knowledge_fingerprint=knowledge_fingerprint,
        )
        if cache_hit is None:
            cache_hit = semantic_cache_check(db, rewritten["semantic_key"], intent, knowledge_fingerprint=knowledge_fingerprint)
    if cache_hit:
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

    if _uses_direct_support_lane(intent, safety):
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
        "safety_gate_ms": round((t_safety - started) * 1000, 2),
        "intent_routing_ms": round((t_routing - t_safety) * 1000, 2),
        "retrieval_ms": round((t_retrieval - t_routing) * 1000, 2),
        "rerank_ms": round((t_rerank - t_retrieval) * 1000, 2),
        "generation_ms": round((t_generation - t_rerank) * 1000, 2),
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


# input_guardrail_check moved to backend.services.agent_input_gate as
# part of the agent_rag.py module split.  Re-exported below.
from backend.services.agent_input_gate import input_guardrail_check  # noqa: F401, E402


# safety_scope_check moved to backend.services.agent_safety as part of the
# agent_rag.py module split.  Re-exported here so existing imports
# (chat, eval scripts, tests) keep working unchanged.
from backend.services.agent_safety import safety_scope_check  # noqa: F401, E402


# route_intent + the three conversation detectors moved to
# backend.services.agent_intent_router.  Re-exported so existing imports
# keep working.
from backend.services.agent_intent_router import (  # noqa: F401, E402
    _is_conversation_opening,
    _is_identity_or_capability_question,
    _is_social_checkin,
    route_intent,
)


def _validated_preselected_intent(intent, safety):
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


# _uses_direct_support_lane moved to backend.services.agent_answer_composition
# as part of the agent_rag.py module split.  Re-imported below alongside
# the rest of the answer-composition module.


# rewrite_and_decompose moved to backend.services.agent_query_rewriting as
# part of the agent_rag.py module split.  Re-exported so existing imports
# keep working.
from backend.services.agent_query_rewriting import rewrite_and_decompose  # noqa: F401, E402


# exact_cache_check + semantic_cache_check moved to
# backend.services.agent_cache as part of the agent_rag.py module split.
# Re-imported below alongside the rest of the cache layer.


# hybrid_retrieval + expand_parent_child_windows moved to
# backend.services.agent_retrieval as part of the agent_rag.py module
# split.  Re-imported below alongside the rest of the retrieval module.


# KB corpus loader (knowledge_snippets / invalidate_kb_cache /
# get_rag_corpus / knowledge_base_fingerprint) moved to
# backend.services.agent_kb_corpus.  Re-exported below so existing
# import sites + the lazy imports inside agent_retrieval / agent_cache
# keep working.
from backend.services.agent_kb_corpus import (  # noqa: F401, E402
    _invalidate_kb_cache,
    _knowledge_snippets,
    get_rag_corpus,
    knowledge_base_fingerprint,
)


# rerank_context, contextual_compression, _CURATED_SOURCES,
# _cross_encoder_* helpers moved to backend.services.agent_retrieval as
# part of the agent_rag.py module split.  Re-imported below alongside
# the rest of the retrieval module.
from backend.services.agent_retrieval import (  # noqa: F401, E402
    CURATED_SOURCES as _CURATED_SOURCES,
    MAX_CONTEXT_CHARS,
    _cross_encoder_enabled,
    _cross_encoder_scores,
    _get_cross_encoder,
    _reranker_backend,
    contextual_compression,
    expand_parent_child_windows,
    hybrid_retrieval,
    rerank_context,
)


# generate_answer, validate_answer_and_citations, REFUSAL_INTENTS, and
# their helpers moved to backend.services.agent_answer_composition.
# Re-imported so existing imports + the few in-module references keep
# working.
from backend.services.agent_answer_composition import (  # noqa: F401, E402
    REFUSAL_INTENTS,
    _uses_direct_support_lane,
    generate_answer,
    validate_answer_and_citations,
)


# _apply_post_gen_validator + _apply_intent_aware_rag_layer moved to
# backend.services.agent_post_gen as part of the agent_rag.py module
# split.  Re-imported so _finalize_result keeps calling them by name.
from backend.services.agent_post_gen import (  # noqa: F401, E402
    _apply_intent_aware_rag_layer,
    _apply_post_gen_validator,
)


def _finalize_result(db, patient_id, query, rewritten, result, retrieved, reranked, compressed, input_guardrails, started):
    """Orchestrate the post-generation pipeline:

      1. Compute latency.
      2. Run legacy output-guardrail heuristics.
      3. Run the post-gen safety validator (may rewrite the reply).
      4. Run the intent-aware RAG layer (mode → tier filter → claim
         validation → evidence grade → optional insufficient-evidence
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
        "input": input_guardrails,
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


# output_guardrail_check moved to backend.services.agent_output_gate
# (sibling of agent_input_gate).  Re-exported here for back-compat.
from backend.services.agent_output_gate import output_guardrail_check  # noqa: F401, E402


# evaluate_rag_response + the per-metric scorers moved to
# backend.services.agent_eval_scoring as part of the agent_rag.py module
# split.  Re-imported below.


# _store_rag_evaluation_log moved to backend.services.agent_eval_log
# (re-exported below).
from backend.services.agent_eval_log import _store_rag_evaluation_log  # noqa: F401, E402


# _contains_diagnostic_or_treatment_claim moved to
# backend.services.agent_answer_composition (re-imported above).

# _content_tokens / _estimate_tokens / _score_status / _cost_latency_note
# moved to backend.services.agent_eval_scoring.  Re-imported below.
from backend.services.agent_eval_scoring import (  # noqa: F401, E402
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


# is_cacheable + store_cache + cache policy/freshness helpers moved to
# backend.services.agent_cache (re-imported below alongside the lookups).


# _safety_reply moved to backend.services.agent_answer_composition
# (re-imported via _safety_reply alias below for back-compat).
from backend.services.agent_answer_composition import _safety_reply  # noqa: F401, E402


# _security_block_reply moved to backend.services.agent_input_gate.
from backend.services.agent_input_gate import _security_block_reply  # noqa: F401, E402


# _with_related_guidance / _educational_reply / _educational_query_bridge /
# _clean_context_text / _should_include_supporting_context moved to
# backend.services.agent_answer_composition.  agent_rag doesn't reference
# them directly anymore — the answer-composition module owns the full
# educational-reply pipeline.


# _intent_boost / _domain_boost / _section_boost moved to
# backend.services.agent_retrieval (re-imported via the agent_retrieval
# import block earlier in this module).
from backend.services.agent_retrieval import (  # noqa: F401, E402
    _domain_boost,
    _intent_boost,
    _section_boost,
)


# _mark_cache_hit + _cache_rejection_reason moved to
# backend.services.agent_cache (re-imported below).


# _trace moved to backend.services.agent_trace as part of the
# agent_rag.py module split.  Re-exported so existing imports keep
# working.
from backend.services.agent_trace import _trace  # noqa: F401, E402


# _semantic_key, _normalize_query, _tokenize moved to
# backend.services.agent_query_rewriting.  Re-imported below so the ~15
# internal call sites in this module keep resolving via the same names.
from backend.services.agent_query_rewriting import (  # noqa: E402
    _normalize_query,
    _semantic_key,
    _tokenize,
)


# Cache layer + cache-adjacent utilities now live in
# backend.services.agent_cache.  Re-import the full surface so existing
# in-module references AND external callers via agent_rag keep working.
from backend.services.agent_cache import (  # noqa: F401, E402
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
