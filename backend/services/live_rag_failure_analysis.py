from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_PATH = "Data/evals/rag/latest_live_rag_failure_analysis.json"


def build_live_rag_failure_analysis(
    *,
    live_path: str = "Data/evals/rag/latest_live_rag_eval.json",
    tier_path: str = "Data/evals/rag/latest_rag_tier_ablation.json",
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    live = _load(live_path)
    tier = _load(tier_path)
    rows: list[dict[str, Any]] = []
    rows.extend(_analyze_cases("live_agent", live.get("cases") or []))

    for config in tier.get("per_config") or []:
        config_name = config.get("config")
        sub_path = Path(tier_path).parent / f"_tier_{config_name}_eval.json"
        sub = _load(str(sub_path))
        rows.extend(_analyze_cases(f"tier_ablation:{config_name}", sub.get("cases") or []))

    counts = Counter(row["failure_category"] for row in rows)
    pass_rates = {
        "live_agent": _dig(live, ["summary", "pass_rate"]),
        **{
            str(item.get("config")): item.get("pass_rate")
            for item in tier.get("per_config") or []
        },
    }
    payload = {
        "schema_version": "live_rag_failure_analysis_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if not rows else "needs_attention",
        "summary": {
            "failure_count": len(rows),
            "failure_categories": dict(counts),
            "pass_rates": pass_rates,
            "unsafe_answer_rate_live": _dig(live, ["summary", "unsafe_answer_rate"]),
            "source_tier_correctness_live": _dig(live, ["summary", "source_tier_correctness"]),
        },
        "failures": rows,
        "recommendation": _recommendation(counts),
        "claim_boundary": (
            "Failure analysis is engineering triage over curated RAG eval artifacts. "
            "It does not prove medical correctness or clinical safety."
        ),
    }
    _write_json(output_path, payload)
    return payload


def _analyze_cases(scope: str, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        if case.get("passed"):
            continue
        rows.append({
            "scope": scope,
            "case_id": case.get("case_id"),
            "query": case.get("query"),
            "failure_category": _category(case),
            "expected_intent": case.get("expected_intent"),
            "observed_intent": case.get("observed_intent"),
            "expected_mode": case.get("expected_mode"),
            "observed_mode": case.get("observed_mode"),
            "citation_status": case.get("citation_status"),
            "claim_support_rate": case.get("claim_support_rate"),
            "tier_correctness": case.get("tier_correctness"),
            "refusal_correctness": case.get("refusal_correctness"),
            "escalation_correctness": case.get("escalation_correctness"),
            "unsafe_blocked": case.get("unsafe_blocked"),
            "grade": case.get("grade"),
        })
    return rows


def _category(case: dict[str, Any]) -> str:
    if case.get("unsafe_blocked"):
        return "unsafe_answer"
    if case.get("tier_correctness") is False:
        return "source_tier_filtering"
    if case.get("observed_intent") != case.get("expected_intent"):
        return "intent_routing_error"
    if case.get("refusal_correctness") is False:
        if case.get("expected_intent") in {"safety_boundary", "treatment_decision_boundary"}:
            return "under_refusal"
        return "over_refusal_or_escalation_gap"
    if case.get("escalation_correctness") is False:
        return "escalation_wording_gap"
    if case.get("citation_status") in {"unsupported", "partial"}:
        return "claim_validation_or_citation_gap"
    if not case.get("cited_sources") and case.get("expected_mode"):
        return "citation_assembly_failure"
    return "other"


def _recommendation(counts: Counter) -> list[str]:
    recs = []
    if counts.get("intent_routing_error"):
        recs.append("Tighten pre-generation medical-boundary routing before retrieval.")
    if counts.get("under_refusal"):
        recs.append("Add boundary synonyms/templates so treatment, diagnosis, tumor-marker, and genetics requests refuse before RAG generation.")
    if counts.get("over_refusal_or_escalation_gap"):
        recs.append("Separate safe educational questions from patient-specific decision requests to reduce over-refusal.")
    if counts.get("claim_validation_or_citation_gap"):
        recs.append("Improve source snippets, parent context recovery, and claim-to-source matching.")
    if not recs:
        recs.append("No current failing live-agent cases; keep unsafe_answer_rate and source-tier correctness as hard blockers.")
    return recs


def _load(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _dig(payload: Any, path: list[Any]) -> Any:
    value = payload
    for key in path:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    return value


def _write_json(path: str, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

