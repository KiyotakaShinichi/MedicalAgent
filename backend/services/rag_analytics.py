import json
from statistics import mean

from backend.models import RAGEvaluationLog


def build_rag_evaluation_summary(db, limit=10):
    rows = (
        db.query(RAGEvaluationLog)
        .order_by(RAGEvaluationLog.created_at.desc(), RAGEvaluationLog.id.desc())
        .all()
    )
    if not rows:
        return {
            "status": "unavailable",
            "purpose": "RAG quality, guardrail, grounding, hallucination, and cost/latency telemetry.",
            "call_count": 0,
            "metric_definitions": metric_definitions(),
            "recent_calls": [],
        }

    precision_values = _numeric_values(row.retrieval_precision_at_3 for row in rows)
    grounding_values = _numeric_values(row.grounding_score for row in rows)
    hallucination_values = _numeric_values(row.hallucination_score for row in rows)
    latency_values = _numeric_values(row.latency_ms for row in rows)
    total_token_values = _numeric_values(row.estimated_total_tokens for row in rows)
    cache_hits = [row for row in rows if row.cache_status in {"exact_cache_hit", "semantic_cache_hit"}]

    return {
        "status": _overall_status(rows),
        "purpose": "RAG quality, guardrail, grounding, hallucination, and cost/latency telemetry.",
        "call_count": len(rows),
        "cache_hit_rate": round(len(cache_hits) / len(rows), 3),
        "average_retrieval_precision_at_3": _round_mean(precision_values),
        "average_grounding_score": _round_mean(grounding_values),
        "average_hallucination_score": _round_mean(hallucination_values),
        "hallucination_risk_counts": _counts(row.hallucination_risk or "unknown" for row in rows),
        "input_guardrail_counts": _counts(row.input_guardrail_status or "unknown" for row in rows),
        "output_guardrail_counts": _counts(row.output_guardrail_status or "unknown" for row in rows),
        "average_latency_ms": _round_mean(latency_values),
        "p95_latency_ms": _percentile(latency_values, 0.95),
        "average_estimated_total_tokens": _round_mean(total_token_values),
        "estimated_llm_cost_usd": round(sum(float(row.estimated_llm_cost_usd or 0) for row in rows), 6),
        "api_costs": _api_cost_summary(rows),
        "cost_latency_tradeoffs": _cost_latency_tradeoffs(rows),
        "metric_definitions": metric_definitions(),
        "recent_calls": [_row_to_dict(row) for row in rows[:limit]],
        "limitations": (
            "These are lightweight production proxies. Use RAGAS/context labels later for formal context precision, answer relevancy, and faithfulness."
        ),
    }


def metric_definitions():
    return [
        {
            "metric": "retrieval_precision_at_3",
            "appropriate_now": True,
            "current_method": "Heuristic source relevance from query-token overlap in top-3 retrieved/reranked items.",
            "later_upgrade": "Replace with labeled precision@k and RAGAS context precision once research-paper KB labels exist.",
        },
        {
            "metric": "answer_grounding_score",
            "appropriate_now": True,
            "current_method": "Content-token overlap between answer and retrieved context.",
            "later_upgrade": "Use RAGAS faithfulness/answer relevancy and/or LLM-as-judge with citations.",
        },
        {
            "metric": "hallucination_score",
            "appropriate_now": True,
            "current_method": "Inverse grounding plus citation and guardrail penalties.",
            "later_upgrade": "Use RAGAS faithfulness plus clinician/SME review labels.",
        },
        {
            "metric": "guardrail_pass_rate",
            "appropriate_now": True,
            "current_method": "Input/output guardrail status from safety, privacy, prompt-injection, diagnosis, treatment-directive, and citation checks.",
            "later_upgrade": "Add red-team test sets and policy regression suites.",
        },
        {
            "metric": "latency_and_estimated_cost",
            "appropriate_now": True,
            "current_method": "Per-call latency and estimated tokens; current LLM cost is zero for deterministic path.",
            "later_upgrade": "Track provider-level token/cost telemetry when LLM generation or RAGAS judges are enabled.",
        },
        {
            "metric": "user_feedback_rating",
            "appropriate_now": True,
            "current_method": "Patient thumbs-up and 1-5 rating on assistant answers.",
            "later_upgrade": "Pair with clinician review and issue taxonomy.",
        },
    ]


def _api_cost_summary(rows):
    total_input_tokens = sum(int(row.estimated_input_tokens or 0) for row in rows)
    total_output_tokens = sum(int(row.estimated_output_tokens or 0) for row in rows)
    total_tokens = sum(int(row.estimated_total_tokens or 0) for row in rows)
    total_cost = round(sum(float(row.estimated_llm_cost_usd or 0) for row in rows), 6)
    generated_calls = [row for row in rows if row.cache_status not in {"exact_cache_hit", "semantic_cache_hit"}]
    cache_hits = [row for row in rows if row.cache_status in {"exact_cache_hit", "semantic_cache_hit"}]
    return {
        "total_estimated_api_cost_usd": total_cost,
        "average_estimated_api_cost_usd": round(total_cost / len(rows), 6) if rows else 0.0,
        "estimated_input_tokens": total_input_tokens,
        "estimated_output_tokens": total_output_tokens,
        "estimated_total_tokens": total_tokens,
        "generated_call_count": len(generated_calls),
        "cache_hit_count": len(cache_hits),
        "cost_basis": "Current RAG answer path is deterministic/local, so provider API cost is logged as $0. Token estimates are ready for Groq/OpenAI/RAGAS judge costs later.",
        "admin_note": "Use this card to compare cache hits, generated calls, latency, and future provider token costs.",
    }


def _cost_latency_tradeoffs(rows):
    groups = {}
    for row in rows:
        key = row.cache_status or "unknown"
        groups.setdefault(key, []).append(row)
    output = []
    for key, items in sorted(groups.items()):
        latencies = _numeric_values(item.latency_ms for item in items)
        tokens = _numeric_values(item.estimated_total_tokens for item in items)
        output.append({
            "path": key,
            "count": len(items),
            "average_latency_ms": _round_mean(latencies),
            "average_estimated_total_tokens": _round_mean(tokens),
            "meaning": _path_meaning(key),
        })
    return output


def _row_to_dict(row):
    return {
        "id": row.id,
        "patient_id": row.patient_id,
        "intent": row.intent,
        "safety_level": row.safety_level,
        "cache_status": row.cache_status,
        "retrieval_precision_at_3": row.retrieval_precision_at_3,
        "grounding_score": row.grounding_score,
        "hallucination_score": row.hallucination_score,
        "hallucination_risk": row.hallucination_risk,
        "input_guardrail_status": row.input_guardrail_status,
        "output_guardrail_status": row.output_guardrail_status,
        "latency_ms": row.latency_ms,
        "estimated_total_tokens": row.estimated_total_tokens,
        "retrieved_source_ids": _loads(row.retrieved_source_ids_json),
        "cited_source_ids": _loads(row.cited_source_ids_json),
        "guardrail_issues": _loads(row.guardrail_issues_json),
        "created_at": str(row.created_at),
    }


def _overall_status(rows):
    output_failures = sum(1 for row in rows if row.output_guardrail_status == "failed")
    high_hallucination = sum(1 for row in rows if row.hallucination_risk == "high")
    if output_failures:
        return "failed"
    if high_hallucination:
        return "unideal"
    average_grounding = _round_mean(_numeric_values(row.grounding_score for row in rows))
    if average_grounding is not None and average_grounding >= 0.55:
        return "strong"
    return "acceptable"


def _path_meaning(path):
    if path in {"exact_cache_hit", "semantic_cache_hit"}:
        return "Reused a validated low-risk answer; usually lower latency and generation cost."
    if path == "stored":
        return "Generated and stored a validated low-risk answer for future reuse."
    if path == "not_cacheable":
        return "Generated but not cached due to safety, patient-specific, or data-entry constraints."
    return "Pipeline path not categorized."


def _numeric_values(values):
    return [float(value) for value in values if value is not None]


def _round_mean(values):
    return round(mean(values), 3) if values else None


def _percentile(values, percentile):
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * percentile))))
    return round(values[index], 3)


def _counts(values):
    output = {}
    for value in values:
        output[value] = output.get(value, 0) + 1
    return output


def _loads(value):
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None
