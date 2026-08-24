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

# `x as x` and the noqa mark these as deliberate re-exports. They were module
# attributes of agent_rag before the split; four earlier refactors in this
# repository broke consumers by dropping names that were only ever imported
# here, so the facade keeps the whole surface rather than the subset it uses.
from collections.abc import Mapping as Mapping  # noqa: F401
from datetime import datetime as datetime, timezone as timezone  # noqa: F401
from time import perf_counter as perf_counter  # noqa: F401
from types import SimpleNamespace as SimpleNamespace  # noqa: F401

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
from backend.services.rag_evidence_envelope import (  # noqa: F401
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

# ─── Extracted orchestration ────────────────────────────────────────────────
#
# Branch execution and response shaping now live in their own modules. They are
# imported here rather than called through their modules because the pipeline's
# long-standing test contract is that patching an attribute on THIS module is
# authoritative - `monkeypatch.setattr(agent_rag, "_run_patient_agent_pipeline_impl", ...)`
# must change what `run_patient_agent_pipeline` calls. Binding the name into
# this namespace and calling it unqualified is what preserves that; importing
# the module and calling `agent_pipeline_runner._run_...` would not.
#
# This is the same indirection already documented above for
# `route_intent_with_local_llm`.
from backend.services.agent_cache import _lookup_cache  # noqa: E402,F401
from backend.services.agent_intent_router import (  # noqa: E402,F401
    _validated_preselected_intent,
)
from backend.services.agent_pipeline_runner import (  # noqa: E402,F401
    _run_cache_hit_branch,
    _run_direct_support_branch,
    _run_input_guardrail_block_branch,
    _run_patient_agent_pipeline_impl,
    _run_rag_generation_branch,
)
from backend.services.agent_result_shaping import (  # noqa: E402,F401
    _attach_turn_trace,
    _finalize_result,
    _prefer_stricter_safety,
    _trace_model_label,
    _trace_refusal_reason,
    _trace_refused,
    _trace_route_reason,
    _trace_usage_cost,
)


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
