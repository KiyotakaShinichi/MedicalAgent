"""RAG-response evaluation scoring for the patient agent.

Every agent response carries an ``rag_evaluation`` envelope produced by
:func:`evaluate_rag_response`.  It bundles four engineering proxies:

  - retrieval precision @ 3 (token overlap with retrieved sources)
  - answer grounding (content-token overlap between reply and context)
  - hallucination risk (inverse grounding + citation + guardrail penalty)
  - cost / latency estimate (chars-to-tokens, no external billing call)

Plus an opt-in LLM-as-judge second opinion mounted under
``answer_grounding_v2_llm_judge`` when ``LLM_JUDGE_ENABLED`` is set.

Every metric here is a **heuristic engineering proxy** — useful for
regression detection across PoC iterations, not for clinical
validation.  The "metric_limitations" string in the envelope says so
out loud.

Extracted from ``agent_rag.py`` as part of the agent_rag.py module
split.  All public functions are re-exported from
``backend.services.agent_rag`` so existing imports work unchanged.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from backend.services.agent_query_rewriting import tokenize


# ─── Constants ───────────────────────────────────────────────────────────────


# Generic, low-signal tokens stripped from content-overlap scoring so
# answer_grounding doesn't reward replies that just echo boilerplate.
# Promoted from an inline local set so reviewers can audit the
# "low-signal" vocabulary at a glance.
GENERIC_CONTENT_TOKENS: frozenset[str] = frozenset({
    "general", "information", "portal", "patient", "team", "care", "use",
    "discuss", "personal", "decisions", "oncology", "medical", "review",
    "contact", "emergency", "services", "support", "assistant",
    "tracking", "education", "only",
})


# Thresholds for the cost/latency tradeoff note.  Picked so a 1-2 chunk
# educational reply at 300-500 tokens lands in "within budget"; a
# multi-chunk synthesis at 800+ tokens or a >1.5s generated path
# triggers the "consider caching" hint.
_COST_TOKEN_THRESHOLD: int = 800
_COST_LATENCY_MS_THRESHOLD: int = 1500


# ─── Low-level scoring helpers ───────────────────────────────────────────────


def content_tokens(text: str) -> list[str]:
    """Tokenize ``text`` and drop generic low-signal tokens.  Used by
    :func:`answer_grounding_score` so generic care-team / oncology
    wording doesn't inflate grounding."""
    return [token for token in tokenize(text) if token not in GENERIC_CONTENT_TOKENS and len(token) > 2]


def estimate_tokens(text: str) -> int:
    """Char-count to token-count approximation (~4 chars/token).  No
    external API call — this is for engineering instrumentation only."""
    return max(1, int(len(text or "") / 4))


def score_status(value: float | None, strong: float, acceptable: float) -> str:
    """Three-tier banding used by all the per-metric envelopes:
    ``strong`` / ``acceptable`` / ``unideal`` / ``unavailable``."""
    if value is None:
        return "unavailable"
    if value >= strong:
        return "strong"
    if value >= acceptable:
        return "acceptable"
    return "unideal"


def cost_latency_note(cache_status: str | None, latency_ms: float, total_tokens: int) -> str:
    """Short human-readable tradeoff note attached to the cost/latency
    envelope."""
    if cache_status in {"exact_cache_hit", "semantic_cache_hit"}:
        return "Cache hit: lower latency and no new retrieval/generation cost."
    if total_tokens > _COST_TOKEN_THRESHOLD or latency_ms > _COST_LATENCY_MS_THRESHOLD:
        return "Generated path is heavier; consider caching if this is low-risk and reusable."
    return "Generated path is within current PoC latency/token budget."


# Underscore aliases preserve the agent_rag in-module call sites.
_content_tokens = content_tokens
_estimate_tokens = estimate_tokens
_score_status = score_status
_cost_latency_note = cost_latency_note


# ─── Per-metric scorers ──────────────────────────────────────────────────────


def proxy_retrieval_precision_at_k(
    items: Iterable[Mapping[str, Any]] | None,
    rewritten: Mapping[str, Any],
    k: int = 3,
) -> dict[str, Any]:
    """Fraction of top-k retrieved items whose title/text/tags share at
    least one expanded-query token.  Replace with labeled precision@k or
    RAGAS context-precision when real labels exist."""
    top = list(items or [])[:k]
    if not top:
        return {
            "metric": "proxy_retrieval_precision_at_3",
            "value": None,
            "k": k,
            "relevant_count": 0,
            "method": "No retrieved context.",
            "status": "unavailable",
        }
    query_tokens = set(tokenize(rewritten.get("expanded_query") or ""))
    relevant_count = 0
    for item in top:
        item_tokens = set(tokenize(" ".join([
            item.get("title", ""),
            item.get("text", ""),
            " ".join(item.get("tags", [])),
        ])))
        if query_tokens & item_tokens:
            relevant_count += 1
    value = round(relevant_count / len(top), 3)
    return {
        "metric": "proxy_retrieval_precision_at_3",
        "value": value,
        "k": len(top),
        "relevant_count": relevant_count,
        "method": (
            "Heuristic query-token overlap with retrieved source title/tags/text. "
            "Replace with labeled precision@k or RAGAS context precision later."
        ),
        "status": score_status(value, strong=0.8, acceptable=0.6),
    }


def answer_grounding_score(reply: str, compressed: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Heuristic content-token overlap between answer and retrieved
    context.  Upgrade to RAGAS faithfulness / answer relevancy later."""
    if not reply:
        return {"score": 0.0, "status": "failed", "method": "Empty reply."}
    if not compressed:
        return {
            "score": None,
            "status": "unavailable",
            "method": "No retrieved context; answer may be deterministic fallback rather than RAG-grounded.",
        }
    reply_tokens = set(content_tokens(reply))
    context_tokens: set[str] = set()
    for item in compressed:
        context_tokens.update(content_tokens(item.get("text", "")))
        context_tokens.update(content_tokens(item.get("title", "")))
    if not reply_tokens:
        score = 0.0
    else:
        score = len(reply_tokens & context_tokens) / len(reply_tokens)
    score = round(score, 3)
    return {
        "score": score,
        "status": score_status(score, strong=0.55, acceptable=0.35),
        "method": (
            "Heuristic content-token overlap between answer and retrieved context. "
            "Upgrade to RAGAS faithfulness/answer relevancy later."
        ),
    }


def hallucination_score(
    grounding_score: float | None,
    validation: Mapping[str, Any],
    input_guardrails: Mapping[str, Any],
    output_guardrails: Mapping[str, Any],
    citations: list,
    compressed: list,
) -> dict[str, Any]:
    """Heuristic inverse grounding + citation/guardrail penalties.
    Replace / compare with RAGAS faithfulness later."""
    issues = set(validation.get("issues") or [])
    issues.update(input_guardrails.get("issues") or [])
    issues.update(output_guardrails.get("issues") or [])
    if grounding_score is None:
        base = 0.25 if not compressed else 0.5
    else:
        base = max(0.0, 1.0 - grounding_score)
    if compressed and not citations:
        base += 0.25
    if issues:
        base += min(0.45, 0.15 * len(issues))
    score = round(min(1.0, base), 3)
    if score <= 0.35:
        risk = "low"
    elif score <= 0.65:
        risk = "medium"
    else:
        risk = "high"
    return {
        "score":  score,
        "risk":   risk,
        "method": (
            "Heuristic inverse grounding plus citation and guardrail penalties. "
            "Replace/compare with RAGAS faithfulness later."
        ),
        "issues": sorted(issues),
    }


def estimate_token_and_cost(
    query: str,
    reply: str,
    compressed: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Estimate input + output token counts (chars/4 heuristic) and
    record cost as 0 — the current agent path is deterministic/local."""
    context_chars = sum(len(item.get("text", "")) for item in compressed)
    input_tokens = estimate_tokens(query) + estimate_tokens(" ".join(item.get("text", "") for item in compressed))
    output_tokens = estimate_tokens(reply)
    total_tokens = input_tokens + output_tokens
    return {
        "estimated_input_tokens":  input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_total_tokens":  total_tokens,
        "estimated_context_chars": context_chars,
        "estimated_llm_cost_usd":  0.0,
        "cost_basis": "Current agent path is deterministic/local. Token estimates are logged for future LLM/RAGAS cost analysis.",
    }


def _maybe_run_llm_judge(
    query: str,
    reply: str,
    compressed: list[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Run the LLM judge if enabled.  Returns None when disabled so the
    eval payload stays small in default runs.  Imports are local because
    the judge is opt-in and we don't want to pull in groq on import."""
    try:
        from backend.services.llm_judge import is_judge_enabled, judge_rag_answer
    except Exception:
        return None
    if not is_judge_enabled():
        return None
    try:
        return judge_rag_answer(
            question=query,
            answer=reply,
            context_chunks=compressed,
        )
    except Exception as exc:  # noqa: BLE001 — never break the pipeline on judge failure
        return {
            "status": "not_computed",
            "reason": f"judge_unexpected_failure: {exc.__class__.__name__}",
            "method": "llm_judge (engineering proxy)",
        }


# ─── Top-level entry point ───────────────────────────────────────────────────


def evaluate_rag_response(
    query: str,
    rewritten: Mapping[str, Any],
    result: Mapping[str, Any],
    retrieved: list[Mapping[str, Any]],
    reranked: list[Mapping[str, Any]],
    compressed: list[Mapping[str, Any]],
    input_guardrails: Mapping[str, Any],
    output_guardrails: Mapping[str, Any],
    latency_ms: float,
) -> dict[str, Any]:
    """Bundle the four engineering proxies into a single envelope.

    The heuristic ``answer_grounding`` and ``hallucination`` are
    metric_v1.  When ``LLM_JUDGE_ENABLED`` is set, an opt-in second
    opinion appears under ``answer_grounding_v2_llm_judge`` so a
    reviewer can see both side-by-side instead of a silent swap.
    """
    retrieval_precision = proxy_retrieval_precision_at_k(reranked or retrieved, rewritten, k=3)
    grounding = answer_grounding_score(result.get("reply") or "", compressed)
    hallucination = hallucination_score(
        grounding_score=grounding["score"],
        validation=result.get("validation") or {},
        input_guardrails=input_guardrails,
        output_guardrails=output_guardrails,
        citations=result.get("citations") or [],
        compressed=compressed,
    )
    judge_result = _maybe_run_llm_judge(query, result.get("reply") or "", compressed)
    token_cost = estimate_token_and_cost(query, result.get("reply") or "", compressed)

    payload: dict[str, Any] = {
        "retrieval_precision_at_3": retrieval_precision,
        "answer_grounding":         grounding,
        "hallucination":            hallucination,
        "cost_latency": {
            **token_cost,
            "latency_ms":   latency_ms,
            "cache_status": (result.get("cache") or {}).get("status"),
            "tradeoff_note": cost_latency_note(
                (result.get("cache") or {}).get("status"),
                latency_ms,
                token_cost["estimated_total_tokens"],
            ),
        },
        "guardrail_summary": {
            "input_status":  input_guardrails.get("status"),
            "output_status": output_guardrails.get("status"),
            "input_issues":  input_guardrails.get("issues") or [],
            "output_issues": output_guardrails.get("issues") or [],
        },
        "metric_limitations": (
            "Retrieval precision and `answer_grounding`/`hallucination` are heuristic token-overlap proxies. "
            "`answer_grounding_v2_llm_judge`, when present, is an LLM-as-judge second opinion — also a proxy, "
            "not clinical validation."
        ),
    }
    if judge_result is not None:
        payload["answer_grounding_v2_llm_judge"] = judge_result
    return payload


__all__ = [
    "GENERIC_CONTENT_TOKENS",
    "content_tokens", "_content_tokens",
    "estimate_tokens", "_estimate_tokens",
    "score_status", "_score_status",
    "cost_latency_note", "_cost_latency_note",
    "proxy_retrieval_precision_at_k",
    "answer_grounding_score",
    "hallucination_score",
    "estimate_token_and_cost",
    "evaluate_rag_response",
    "_maybe_run_llm_judge",
]
