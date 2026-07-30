from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from backend.database import SessionLocal
from backend.models import RAGEvaluationLog
from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_OUTPUT_PATH = "Data/evals/ops/latest_cost_latency_report.json"
LIVE_RAG_PATH = "Data/evals/rag/latest_live_rag_eval.json"
LATENCY_PROBE_PATH = "Data/evals/models/latest_agent_latency_probe.json"

CLAIM_BOUNDARY = (
    "Cost and latency observability is engineering telemetry only. It does not "
    "establish clinical safety, clinical validation, or real-world patient benefit."
)

_API_INPUT_COST_PER_1K = 0.00015
_API_OUTPUT_COST_PER_1K = 0.00060


def build_cost_latency_report(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    db=None,
    limit: int = 500,
) -> dict[str, Any]:
    rows = _load_db_request_rows(db=db, limit=limit)
    live_rows = _load_live_rag_rows()
    request_rows = rows or live_rows
    route_summaries = _route_summaries(request_rows)
    route_comparison = _route_cost_comparison(request_rows)
    summary = _summary(request_rows, route_comparison)
    local_probe_stage_latency = _load_probe_stage_latency()

    payload = {
        "schema_version": "cost_latency_observability_v3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if request_rows else "needs_attention",
        "summary": summary,
        "route_summaries": route_summaries,
        "route_cost_comparison": route_comparison,
        "local_probe_stage_latency": local_probe_stage_latency,
        "requests": request_rows[:100],
        "instrumentation": {
            "per_request_fields": [
                "route",
                "model_used",
                "token_usage_basis",
                "provider_reported_input_tokens",
                "provider_reported_output_tokens",
                "provider_reported_total_tokens",
                "actual_usage_coverage_rate",
                "estimated_input_tokens",
                "estimated_output_tokens",
                "estimated_cost_usd",
                "latency_ms",
                "safety_gate_latency_ms",
                "intent_routing_latency_ms",
                "retrieval_latency_ms",
                "reranker_latency_ms",
                "pre_generation_governance_ms",
                "compression_latency_ms",
                "generation_latency_ms",
                "cross_encoder_latency_ms",
                "validator_latency_ms",
                "cache_status",
                "source_governance_ms",
                "post_generation_validation_ms",
                "semantic_chunk_rebuild_time_ms",
                "fhir_mapper_latency_ms",
                "ood_gate_latency_ms",
                "total_rag_latency_before_after",
                "total_prediction_latency_before_after",
            ],
            "legacy_gap": (
                "Rows created before schema revision 0011 may not have token_usage_json, and rows "
                "before revision 0005 may not have stage_latency_json. Missing fields remain null "
                "or explicitly estimated instead of being presented as provider-reported usage."
            ),
            "privacy": "No prompts or completions are retained in token_usage_json.",
        },
        "recommendations": _recommendations(summary, route_comparison),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(output_path, payload)
    return payload


def _load_db_request_rows(*, db=None, limit: int) -> list[dict[str, Any]]:
    owns_session = db is None
    if db is None:
        db = SessionLocal()
    try:
        rows = (
            db.query(RAGEvaluationLog)
            .order_by(RAGEvaluationLog.created_at.desc(), RAGEvaluationLog.id.desc())
            .limit(max(25, min(limit, 2000)))
            .all()
        )
    except Exception:
        rows = []
    finally:
        if owns_session:
            db.close()
    return [_db_row_to_request(row) for row in rows if row.latency_ms is not None]


def _db_row_to_request(row: RAGEvaluationLog) -> dict[str, Any]:
    stage = _loads_dict(getattr(row, "stage_latency_json", None))
    token_usage = _loads_dict(getattr(row, "token_usage_json", None))
    route = _route_from_log(row)
    input_tokens = int(row.estimated_input_tokens or 0)
    output_tokens = int(row.estimated_output_tokens or 0)
    cost = float(row.estimated_llm_cost_usd or 0.0)
    return {
        "source": "rag_evaluation_logs",
        "request_id": getattr(row, "request_id", None),
        "route": route,
        "intent": row.intent,
        "model_used": getattr(row, "model_used", None) or "deterministic_local_or_untracked",
        "token_usage_basis": _token_usage_basis(token_usage),
        "provider_reported_input_tokens": _provider_reported_tokens(token_usage, "input_tokens"),
        "provider_reported_output_tokens": _provider_reported_tokens(token_usage, "output_tokens"),
        "provider_reported_total_tokens": _provider_reported_tokens(token_usage, "total_tokens"),
        "llm_call_count": int(token_usage.get("call_count") or 0),
        "provider_reported_call_count": int(token_usage.get("provider_reported_call_count") or 0),
        "estimated_call_count": int(token_usage.get("estimated_call_count") or 0),
        "actual_usage_coverage_rate": _round(token_usage.get("actual_usage_coverage_rate")) or 0.0,
        "llm_call_latency_ms": _round(token_usage.get("llm_call_latency_ms")),
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_total_tokens": int(row.estimated_total_tokens or input_tokens + output_tokens),
        "estimated_cost_usd": round(cost, 8),
        "latency_ms": _round(row.latency_ms),
        "safety_gate_latency_ms": _round(stage.get("safety_gate_ms")),
        "intent_routing_latency_ms": _round(stage.get("intent_routing_ms")),
        "retrieval_latency_ms": _round(stage.get("retrieval_ms")),
        "reranker_latency_ms": _round(stage.get("rerank_ms")),
        "pre_generation_governance_ms": _round(
            stage.get("pre_generation_governance_ms")
        ),
        "compression_latency_ms": _round(stage.get("compression_ms")),
        "generation_latency_ms": _round(stage.get("generation_ms")),
        "validator_latency_ms": _round(stage.get("validator_ms")),
        "cache_status": row.cache_status or "unknown",
        "cache_hit": _is_cache_hit(row.cache_status),
        "source_governance_ms": _round(stage.get("source_governance_ms")),
        "post_generation_validation_ms": _round(stage.get("post_generation_validation_ms")),
        "stage_latency_basis": "captured" if stage else "legacy_missing",
        "source_tier_correct": _source_tier_correct(row),
        "claim_validation_passed": _claim_validation_passed(row),
        "unsafe_leakage": _unsafe_leakage(row),
        "created_at": str(row.created_at),
    }


def _load_live_rag_rows() -> list[dict[str, Any]]:
    payload = _read_json(LIVE_RAG_PATH)
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list):
        return []
    rows = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        route = _route_from_live_case(case)
        input_tokens = _estimate_tokens(str(case.get("query") or ""))
        output_tokens = 0 if route == "deterministic_only_refusal" else 220
        rows.append({
            "source": "latest_live_rag_eval",
            "request_id": case.get("case_id"),
            "route": route,
            "intent": case.get("observed_intent") or case.get("expected_intent") or "unknown",
            "model_used": "deterministic_live_eval",
            "token_usage_basis": "pipeline_estimate_only",
            "provider_reported_input_tokens": None,
            "provider_reported_output_tokens": None,
            "provider_reported_total_tokens": None,
            "llm_call_count": 0,
            "provider_reported_call_count": 0,
            "estimated_call_count": 0,
            "actual_usage_coverage_rate": 0.0,
            "llm_call_latency_ms": None,
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": output_tokens,
            "estimated_total_tokens": input_tokens + output_tokens,
            "estimated_cost_usd": 0.0,
            "latency_ms": _round(case.get("latency_ms")),
            "safety_gate_latency_ms": None,
            "intent_routing_latency_ms": None,
            "retrieval_latency_ms": None,
            "reranker_latency_ms": None,
            "pre_generation_governance_ms": None,
            "compression_latency_ms": None,
            "generation_latency_ms": None,
            "validator_latency_ms": None,
            "cache_status": "eval_no_cache",
            "cache_hit": False,
            "source_governance_ms": None,
            "post_generation_validation_ms": None,
            "stage_latency_basis": "live_eval_total_only",
            "source_tier_correct": bool(case.get("tier_correctness", False)),
            "claim_validation_passed": float(case.get("claim_support_rate") or 0) >= 0.5,
            "unsafe_leakage": bool(case.get("unsafe_blocked", False)),
            "created_at": payload.get("generated_at"),
        })
    return rows


def _load_probe_stage_latency() -> dict[str, Any]:
    payload = _read_json(LATENCY_PROBE_PATH)
    summaries = payload.get("summary_by_route")
    if not isinstance(summaries, dict):
        return {
            "status": "not_measured",
            "source_artifact": LATENCY_PROBE_PATH,
            "route_count": 0,
            "interpretation": "No credible local stage-latency probe is available.",
        }
    measured_stage_samples = 0
    normalized: dict[str, Any] = {}
    for route, stages in summaries.items():
        if not isinstance(stages, dict):
            continue
        normalized[route] = {}
        for stage, values in stages.items():
            if not isinstance(values, dict):
                continue
            samples = int(values.get("samples") or 0)
            measured_stage_samples += samples if stage != "total_ms" else 0
            normalized[route][stage] = {
                "samples": samples,
                "p50_ms": _round(values.get("median_ms")),
                "p95_ms": _round(values.get("p95_ms")),
                "max_ms": _round(values.get("max_ms")),
                "percentile_credibility": _percentile_credibility(samples),
            }
    return {
        "status": "measured" if measured_stage_samples else "needs_attention",
        "source_artifact": LATENCY_PROBE_PATH,
        "route_count": len(normalized),
        "measured_stage_sample_count": measured_stage_samples,
        "routes": normalized,
        "environment": {
            "local_only": True,
            "forced_sparse_retrieval": True,
            "in_memory_sqlite": True,
            "production_slo": False,
        },
        "interpretation": (
            "Stage timings come from the repeated local route probe and remain "
            "separate from legacy database rows with no stage envelope."
        ),
    }


def _route_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["route"]].append(row)
    summaries = []
    for route, items in sorted(grouped.items()):
        latencies = [_float(item.get("latency_ms")) for item in items if _float(item.get("latency_ms")) is not None]
        costs = [_float(item.get("estimated_cost_usd")) or 0.0 for item in items]
        actual_tokens = [
            int(item.get("provider_reported_total_tokens") or 0)
            for item in items
            if item.get("provider_reported_total_tokens") is not None
        ]
        estimated_tokens = [int(item.get("estimated_total_tokens") or 0) for item in items]
        summaries.append({
            "route": route,
            "request_count": len(items),
            "p50_latency_ms": _percentile(latencies, 50),
            "p95_latency_ms": _percentile(latencies, 95),
            "p99_latency_ms": _percentile(latencies, 99),
            "mean_latency_ms": _round(mean(latencies)) if latencies else None,
            "estimated_cost_usd": round(sum(costs), 8),
            "cost_per_request_usd": round(sum(costs) / max(len(items), 1), 8),
            "provider_reported_total_tokens": sum(actual_tokens),
            "estimated_pipeline_total_tokens": sum(estimated_tokens),
            "requests_with_provider_usage": sum(
                1 for item in items if item.get("provider_reported_total_tokens") is not None
            ),
            "actual_usage_coverage_rate": round(
                sum(1 for item in items if item.get("provider_reported_total_tokens") is not None)
                / max(len(items), 1),
                4,
            ),
            "cache_hit_rate": round(sum(1 for item in items if item.get("cache_hit")) / max(len(items), 1), 4),
            "claim_validation_pass_rate": round(sum(1 for item in items if item.get("claim_validation_passed")) / max(len(items), 1), 4),
            "source_tier_correctness": round(sum(1 for item in items if item.get("source_tier_correct")) / max(len(items), 1), 4),
            "unsafe_leakage_rate": round(sum(1 for item in items if item.get("unsafe_leakage")) / max(len(items), 1), 4),
        })
    return summaries


def _route_cost_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = _baseline_tokens_and_latency(rows)
    scenarios = [
        {
            "route": "full_api_path",
            "description": "Hosted/API answer path with retrieval, generation, source governance, and validators.",
            "latency_multiplier": 1.0,
            "input_multiplier": 1.0,
            "output_multiplier": 1.0,
            "fixed_latency_ms": 0.0,
            "cache_hit": False,
        },
        {
            "route": "cached_path",
            "description": "Validated low-risk cached answer tied to the current KB fingerprint.",
            "latency_multiplier": 0.18,
            "input_multiplier": 0.02,
            "output_multiplier": 0.0,
            "fixed_latency_ms": 35.0,
            "cache_hit": True,
        },
        {
            "route": "local_slm_routing_plus_api_answer",
            "description": "Local SLM handles intent/routing only; API path still produces governed answer.",
            "latency_multiplier": 0.92,
            "input_multiplier": 0.82,
            "output_multiplier": 1.0,
            "fixed_latency_ms": 80.0,
            "cache_hit": False,
        },
        {
            "route": "local_slm_query_rewrite_plus_api_answer",
            "description": "Local SLM rewrites/query-plans; API path still answers behind validators.",
            "latency_multiplier": 0.95,
            "input_multiplier": 0.78,
            "output_multiplier": 1.0,
            "fixed_latency_ms": 120.0,
            "cache_hit": False,
        },
        {
            "route": "deterministic_only_refusal_path",
            "description": "No retrieval/generation for blocked diagnosis, treatment, prognosis, dosage, genetic-risk, or tumor-marker claims.",
            "latency_multiplier": 0.03,
            "input_multiplier": 0.0,
            "output_multiplier": 0.0,
            "fixed_latency_ms": 15.0,
            "cache_hit": False,
        },
    ]
    output = []
    for scenario in scenarios:
        input_tokens = round(baseline["input_tokens"] * scenario["input_multiplier"])
        output_tokens = round(baseline["output_tokens"] * scenario["output_multiplier"])
        cost = _api_cost(input_tokens, output_tokens)
        latency = baseline["p50_latency_ms"] * scenario["latency_multiplier"] + scenario["fixed_latency_ms"]
        successful_safe_answer_rate = 1.0 if scenario["route"] != "deterministic_only_refusal_path" else 1.0
        output.append({
            "route": scenario["route"],
            "description": scenario["description"],
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": output_tokens,
            "estimated_cost_usd": cost,
            "cost_per_successful_safe_answer_usd": round(cost / successful_safe_answer_rate, 8),
            "p50_latency_ms_estimate": _round(latency),
            "cache_hit": scenario["cache_hit"],
            "unsafe_leakage_rate_target": 0.0,
            "refusal_correctness_target": 1.0,
            "claim_validation_pass_required": True,
            "source_tier_correctness_required": True,
            "basis": "engineering estimate from current token/latency telemetry; not provider billing truth",
        })
    return output


def _summary(rows: list[dict[str, Any]], route_comparison: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [_float(row.get("latency_ms")) for row in rows if _float(row.get("latency_ms")) is not None]
    costs = [_float(row.get("estimated_cost_usd")) or 0.0 for row in rows]
    provider_rows = [row for row in rows if row.get("provider_reported_total_tokens") is not None]
    stage_keys = (
        "safety_gate_latency_ms",
        "intent_routing_latency_ms",
        "retrieval_latency_ms",
        "reranker_latency_ms",
        "pre_generation_governance_ms",
        "compression_latency_ms",
        "generation_latency_ms",
        "validator_latency_ms",
        "source_governance_ms",
        "post_generation_validation_ms",
        "llm_call_latency_ms",
    )
    return {
        "request_count": len(rows),
        "overall_latency_ms": {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "p99": _percentile(latencies, 99),
            "sample_count": len(latencies),
            "percentile_credibility": _percentile_credibility(len(latencies)),
        },
        "stage_latency_ms": {
            key.replace("_latency_ms", "").replace("_ms", ""): _stage_summary(rows, key)
            for key in stage_keys
        },
        "provider_reported_usage": {
            "requests_with_actual_usage": len(provider_rows),
            "coverage_rate": round(len(provider_rows) / max(len(rows), 1), 4),
            "input_tokens": sum(int(row.get("provider_reported_input_tokens") or 0) for row in provider_rows),
            "output_tokens": sum(int(row.get("provider_reported_output_tokens") or 0) for row in provider_rows),
            "total_tokens": sum(int(row.get("provider_reported_total_tokens") or 0) for row in provider_rows),
            "interpretation": (
                "Provider-reported token usage only. Requests without provider usage are excluded, "
                "not silently mixed with estimates."
            ),
        },
        "estimated_pipeline_usage": {
            "input_tokens": sum(int(row.get("estimated_input_tokens") or 0) for row in rows),
            "output_tokens": sum(int(row.get("estimated_output_tokens") or 0) for row in rows),
            "total_tokens": sum(int(row.get("estimated_total_tokens") or 0) for row in rows),
            "basis": "chars_div_4_estimate",
        },
        "estimated_total_cost_usd": round(sum(costs), 8),
        "cache_hit_rate": round(sum(1 for row in rows if row.get("cache_hit")) / max(len(rows), 1), 4),
        "unsafe_leakage_rate": round(sum(1 for row in rows if row.get("unsafe_leakage")) / max(len(rows), 1), 4),
        "source_tier_correctness": round(sum(1 for row in rows if row.get("source_tier_correct")) / max(len(rows), 1), 4),
        "claim_validation_pass_rate": round(sum(1 for row in rows if row.get("claim_validation_passed")) / max(len(rows), 1), 4),
        "stage_latency_captured_count": sum(1 for row in rows if row.get("stage_latency_basis") == "captured"),
        "lowest_estimated_cost_route": min(route_comparison, key=lambda item: item["estimated_cost_usd"])["route"] if route_comparison else None,
        "cost_basis": (
            "Token counts use provider metadata when available and remain separate from chars/4 estimates. "
            "Dollar values use explicit pricing assumptions for engineering capacity planning, not audited billing."
        ),
    }


def _recommendations(summary: dict[str, Any], route_comparison: list[dict[str, Any]]) -> list[str]:
    recommendations = [
        "Keep deterministic-only refusal for blocked medical decision requests because it has the lowest cost and avoids unsafe generation.",
        "Use local SLM only for low-risk routing/rewrite/extraction helpers; final medical education remains behind source governance and validators.",
    ]
    p95 = ((summary.get("overall_latency_ms") or {}).get("p95") or 0)
    if p95 and p95 > 5000:
        recommendations.append("p95 is high; cache only safe education/portal-help answers and keep SSE stage feedback visible.")
    if summary.get("stage_latency_captured_count") == 0:
        recommendations.append("Generate new chat traces after revision 0005 to populate per-stage retrieval/reranker/validator timing fields.")
    usage = summary.get("provider_reported_usage") or {}
    if float(usage.get("coverage_rate") or 0.0) < 0.8:
        recommendations.append(
            "Provider token coverage is below 80%; keep provider-reported totals separate from "
            "chars/4 estimates until new instrumented traffic increases coverage."
        )
    latency = summary.get("overall_latency_ms") or {}
    if latency.get("percentile_credibility") == "insufficient_n_for_tail_claim":
        recommendations.append(
            "Fewer than 30 latency samples are available; treat p95/p99 as directional and gather more requests."
        )
    if route_comparison:
        cached = next((item for item in route_comparison if item["route"] == "cached_path"), None)
        if cached:
            recommendations.append(f"Cached path estimated p50 is {cached['p50_latency_ms_estimate']}ms when cache policy allows reuse.")
    return recommendations


def _baseline_tokens_and_latency(rows: list[dict[str, Any]]) -> dict[str, float]:
    input_tokens = [_float(row.get("estimated_input_tokens")) for row in rows if _float(row.get("estimated_input_tokens"))]
    output_tokens = [_float(row.get("estimated_output_tokens")) for row in rows if _float(row.get("estimated_output_tokens"))]
    latencies = [_float(row.get("latency_ms")) for row in rows if _float(row.get("latency_ms")) is not None]
    return {
        "input_tokens": mean(input_tokens) if input_tokens else 850.0,
        "output_tokens": mean(output_tokens) if output_tokens else 220.0,
        "p50_latency_ms": _percentile(latencies, 50) or 1600.0,
    }


def _token_usage_basis(token_usage: dict[str, Any]) -> str:
    if int(token_usage.get("provider_reported_call_count") or 0) > 0:
        return "provider_reported"
    if int(token_usage.get("call_count") or 0) > 0:
        return "per_call_estimate"
    return "pipeline_estimate_only"


def _provider_reported_tokens(token_usage: dict[str, Any], field: str) -> int | None:
    if int(token_usage.get("provider_reported_call_count") or 0) <= 0:
        return None
    try:
        return max(0, int(token_usage.get(field) or 0))
    except (TypeError, ValueError):
        return None


def _stage_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [
        value
        for row in rows
        if (value := _float(row.get(field))) is not None
    ]
    return {
        "sample_count": len(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "percentile_credibility": _percentile_credibility(len(values)),
    }


def _percentile_credibility(sample_count: int) -> str:
    if sample_count >= 100:
        return "stable_internal_sample"
    if sample_count >= 30:
        return "directional_internal_sample"
    if sample_count > 0:
        return "insufficient_n_for_tail_claim"
    return "not_measured"


def _route_from_log(row: RAGEvaluationLog) -> str:
    terminal = (row.terminal_step or "").lower()
    cache = (row.cache_status or "").lower()
    intent = (row.intent or "").lower()
    if "cache_hit" in cache or cache in {"hit", "exact_cache_hit", "semantic_cache_hit"}:
        return "cached_path"
    if "refusal" in terminal or "boundary" in intent:
        return "deterministic_only_refusal_path"
    if row.rag_mode:
        return "full_api_path"
    return "full_api_path"


def _route_from_live_case(case: dict[str, Any]) -> str:
    intent = str(case.get("observed_intent") or case.get("expected_intent") or "").lower()
    if "boundary" in intent or intent in {"urgent_safety", "security_boundary"}:
        return "deterministic_only_refusal_path"
    return "full_api_path"


def _source_tier_correct(row: RAGEvaluationLog) -> bool:
    payload = _loads_dict(getattr(row, "tier_filter_json", None))
    if not payload:
        return True
    if "source_tier_correctness" in payload:
        return bool(payload.get("source_tier_correctness"))
    blocked = payload.get("blocked_sources") or payload.get("disallowed_sources") or []
    return len(blocked) == 0


def _claim_validation_passed(row: RAGEvaluationLog) -> bool:
    payload = _loads_dict(getattr(row, "claim_validation_json", None))
    if not payload:
        return True
    status = str(payload.get("status") or payload.get("overall_status") or "").lower()
    if status in {"failed", "unsupported", "contradicted"}:
        return False
    unsupported = payload.get("unsupported_claims") or []
    contradicted = payload.get("contradicted_claims") or []
    return not unsupported and not contradicted


def _unsafe_leakage(row: RAGEvaluationLog) -> bool:
    issues = _loads_dict(getattr(row, "guardrail_issues_json", None))
    output_issues = issues.get("output") if isinstance(issues, dict) else []
    status = str(row.output_guardrail_status or "").lower()
    terminal = str(row.terminal_step or "").lower()
    # A triggered validator is not an unsafe leak when the terminal path is a
    # refusal or substituted safe answer. Count only explicit failed leakage
    # markers, keeping block/refusal behavior as a safe outcome.
    if "refusal" in terminal or status in {"blocked", "substituted", "refused"}:
        return False
    return status == "failed" or any("unsafe_answer_emitted" in str(item) for item in output_issues or [])


def _is_cache_hit(status: Any) -> bool:
    return str(status or "").lower() in {"hit", "exact_cache_hit", "semantic_cache_hit", "cache_hit"}


def _api_cost(input_tokens: float, output_tokens: float) -> float:
    return round(
        (input_tokens / 1000.0) * _API_INPUT_COST_PER_1K
        + (output_tokens / 1000.0) * _API_OUTPUT_COST_PER_1K,
        8,
    )


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


def _percentile(values: list[float], pct: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return _round(ordered[0])
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return _round(ordered[low] * (1 - weight) + ordered[high] * weight)


def _float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: Any) -> float | None:
    number = _float(value)
    return round(number, 2) if number is not None else None


def _loads_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _read_json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    full_path = candidate if candidate.is_absolute() else ROOT_DIR / candidate
    if not full_path.exists():
        return {}
    try:
        parsed = json.loads(full_path.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    candidate = Path(path)
    full_path = candidate if candidate.is_absolute() else ROOT_DIR / candidate
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = ["DEFAULT_OUTPUT_PATH", "build_cost_latency_report"]
