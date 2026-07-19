"""Canonical RAG effectiveness-versus-governance trade-off report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASELINE_PATH = Path("Data/evals/rag/latest_rag_baseline_comparison.json")
HOLDOUT_PATH = Path("Data/evals/rag/latest_rag_holdout_baseline_comparison.json")
OUTPUT_PATH = Path("Data/evals/rag/latest_rag_governance_tradeoff.json")


def build_rag_governance_tradeoff() -> dict[str, Any]:
    comparison = _read(BASELINE_PATH)
    holdout = _read(HOLDOUT_PATH)
    configurations = comparison.get("configurations") or {}
    bm25 = ((configurations.get("bm25_only") or {}).get("summary") or {})
    full = ((configurations.get("hybrid_rrf_query_rewrite_parent_child_source_tier") or {}).get("summary") or {})
    best = ((configurations.get("hybrid_rrf_query_rewrite") or {}).get("summary") or {})

    recall_delta = _delta(full.get("recall_at_10"), bm25.get("recall_at_10"))
    tier_delta = _delta(full.get("source_tier_correctness"), bm25.get("source_tier_correctness"))
    latency_ratio = _ratio(full.get("latency_p95_ms"), bm25.get("latency_p95_ms"))
    external_completed = holdout.get("completed") is True and holdout.get("external_author_eval_completed") is True

    return {
        "schema_version": "rag_governance_tradeoff_v1_2026_07",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "needs_attention",
        "internal_goldset_n": comparison.get("total_n"),
        "internal_goldset_only": True,
        "bm25": _core(bm25),
        "best_ungoverned_hybrid": _core(best),
        "full_source_governed_stack": _core(full),
        "tradeoffs": {
            "full_minus_bm25_recall_at_10": recall_delta,
            "full_minus_bm25_source_tier_correctness": tier_delta,
            "full_vs_bm25_latency_p95_ratio": latency_ratio,
            "raw_retrieval_improvement_proven": bool(recall_delta is not None and recall_delta > 0),
            "governance_improvement_observed_internally": bool(tier_delta is not None and tier_delta > 0),
        },
        "external_holdout": {
            "completed": external_completed,
            "status": holdout.get("status", "missing"),
            "reason": holdout.get("reason"),
        },
        "decision": (
            "Retain the source-governed configuration for its allowed-use and source-tier controls, "
            "while reporting that raw retrieval superiority over BM25 is not proven. Do not promote "
            "the experimental citation pruner."
        ),
        "improvement_proven_vs_bm25": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "claim_boundary": (
            "This is internal engineering evidence on a frozen, internally authored goldset. It is not "
            "clinical validation, an external generalisation result, or proof of patient benefit."
        ),
    }


def write_rag_governance_tradeoff(path: str | Path = OUTPUT_PATH) -> dict[str, Any]:
    payload = build_rag_governance_tradeoff()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _core(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: summary.get(key)
        for key in (
            "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10", "citation_precision",
            "claim_support_rate", "unsupported_context_rate", "refusal_correctness",
            "source_tier_correctness", "latency_p50_ms", "latency_p95_ms",
        )
    }


def _delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 6)


def _ratio(left: Any, right: Any) -> float | None:
    if left is None or right in {None, 0}:
        return None
    return round(float(left) / float(right), 4)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


__all__ = ["build_rag_governance_tradeoff", "write_rag_governance_tradeoff"]
