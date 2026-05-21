from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep the default evaluator local and predictable. Dense/cross-encoder model
# loading can be enabled explicitly by setting these env vars before running.
os.environ.setdefault("RAG_FORCE_SPARSE", "true")

from backend.services.agent_query_rewriting import rewrite_and_decompose  # noqa: E402
from backend.services.agent_rag import _knowledge_snippets, knowledge_base_fingerprint  # noqa: E402
from backend.services.cross_encoder_reranker import (  # noqa: E402
    cross_encoder_available,
    rerank_with_cross_encoder,
)
from backend.services.governance_readiness_artifacts import (  # noqa: E402
    write_rag_gold_claim_grounding_cases,
)
from backend.services.rag_vector_index import search_hybrid_index  # noqa: E402


OUTPUT_PATH = ROOT / "Data/evals/rag/latest_reranker_ablation.json"
GOLD_CASE_PATH = ROOT / "Data/evals/rag/gold_claim_grounding_cases.jsonl"
STRATEGIES = ("dense_only", "sparse_only", "hybrid_rrf", "hybrid_rrf_cross_encoder")


def main() -> int:
    if not GOLD_CASE_PATH.exists():
        write_rag_gold_claim_grounding_cases()
    cases = _load_cases(GOLD_CASE_PATH)
    corpus = _knowledge_snippets()
    fingerprint = knowledge_base_fingerprint()
    payload = {
        "schema_version": "retrieval_reranker_ablation_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "release_id": "2026-05-ai-swe-hardening",
        "baseline_version": "gold_claim_grounding_cases_v2_2026_05",
        "cross_encoder_available": cross_encoder_available(),
        "cross_encoder_enabled": _env_enabled("RAG_ENABLE_CROSS_ENCODER"),
        "strategies": {},
        "claim_boundary": (
            "Retrieval metrics are engineering evidence for source finding. They do not "
            "prove medical correctness, clinical safety, or real-world answer quality."
        ),
    }

    for strategy in STRATEGIES:
        payload["strategies"][strategy] = _evaluate_strategy(strategy, cases, corpus, fingerprint)

    before = payload["strategies"]["hybrid_rrf"]["summary"]
    after = payload["strategies"]["hybrid_rrf_cross_encoder"]["summary"]
    payload["summary"] = {
        "case_count": len(cases),
        "before_live_rag_pass_rate_proxy": before["recall_at_10"],
        "after_live_rag_pass_rate_proxy": after["recall_at_10"],
        "before_unsupported_answer_rate": before["unsupported_answer_rate"],
        "after_unsupported_answer_rate": after["unsupported_answer_rate"],
        "before_source_tier_correctness": before["source_tier_correctness"],
        "after_source_tier_correctness": after["source_tier_correctness"],
        "after_mrr": after["mrr"],
        "after_ndcg_at_10": after["ndcg_at_10"],
        "after_recall_at_10": after["recall_at_10"],
        "source_hit_rate": after["source_hit_rate"],
        "citation_support_rate": after["citation_support_rate"],
        "claim_support_rate": after["claim_support_rate"],
        "unsupported_answer_rate": after["unsupported_answer_rate"],
        "p50_retrieval_latency_ms": after["p50_retrieval_latency_ms"],
        "p95_retrieval_latency_ms": after["p95_retrieval_latency_ms"],
        "reranker_latency_ms": after["reranker_latency_ms"],
    }
    payload["status"] = (
        "acceptable"
        if after["unsupported_answer_rate"] <= 0.5 and after["source_tier_correctness"] >= 1.0
        else "needs_attention"
    )
    _write_json(OUTPUT_PATH, payload)
    print(json.dumps(payload["summary"], indent=2))
    return 0


def _evaluate_strategy(
    strategy: str,
    cases: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    fingerprint: str,
) -> dict[str, Any]:
    previous = os.environ.get("RAG_ENABLE_CROSS_ENCODER")
    allow_model_eval = _env_enabled("RAG_EVAL_ENABLE_CROSS_ENCODER")
    if strategy != "hybrid_rrf_cross_encoder" or not allow_model_eval:
        os.environ["RAG_ENABLE_CROSS_ENCODER"] = "false"
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    reranker_latencies: list[float] = []
    try:
        for case in cases:
            started = time.perf_counter()
            intent = str(case.get("expected_intent") or "education")
            rewritten = rewrite_and_decompose(str(case.get("user_query") or ""), intent)
            candidates = search_hybrid_index(
                query=rewritten["expanded_query"],
                corpus=corpus,
                intent=intent,
                knowledge_fingerprint=fingerprint,
                candidate_limit=50,
            )
            if strategy == "dense_only":
                ranked = sorted(candidates, key=lambda row: float(row.get("dense_score") or row.get("vector_score") or 0.0), reverse=True)
            elif strategy == "sparse_only":
                ranked = sorted(candidates, key=lambda row: float(row.get("sparse_score") or row.get("lexical_score") or 0.0), reverse=True)
            elif strategy == "hybrid_rrf_cross_encoder":
                ranked, telemetry = rerank_with_cross_encoder(
                    rewritten["expanded_query"],
                    candidates,
                    top_k=10,
                    candidate_limit=50,
                )
                reranker_latencies.append(float(telemetry.get("reranker_latency_ms") or 0.0))
            else:
                ranked = sorted(candidates, key=lambda row: float(row.get("retrieval_score") or 0.0), reverse=True)
            latency_ms = (time.perf_counter() - started) * 1000
            latencies.append(latency_ms)
            ranked = ranked[:10]
            expected = _expected_ids(case)
            rank = _first_expected_rank(ranked, expected)
            source_tier_ok = all(str(row.get("source_tier") or row.get("tier") or "T1") != "T5" for row in ranked)
            rows.append({
                "case_id": case.get("case_id"),
                "category": case.get("category"),
                "expected_source_ids": sorted(expected),
                "retrieved_source_ids": [sorted(_row_ids(row))[0] for row in ranked if _row_ids(row)],
                "first_relevant_rank": rank,
                "recall_at_5": rank is not None and rank <= 5,
                "recall_at_10": rank is not None and rank <= 10,
                "source_hit": rank is not None,
                "citation_support_proxy": rank is not None,
                "claim_support_proxy": rank is not None,
                "source_tier_correct": source_tier_ok,
                "unsupported_answer_proxy": rank is None,
                "mrr": round(1.0 / rank, 4) if rank else 0.0,
                "ndcg_at_10": round(1.0 / math.log2(rank + 1), 4) if rank else 0.0,
                "latency_ms": round(latency_ms, 3),
            })
    finally:
        if previous is None:
            os.environ.pop("RAG_ENABLE_CROSS_ENCODER", None)
        else:
            os.environ["RAG_ENABLE_CROSS_ENCODER"] = previous
    summary = {
        "case_count": len(rows),
        "mrr": _mean(row["mrr"] for row in rows),
        "ndcg_at_10": _mean(row["ndcg_at_10"] for row in rows),
        "recall_at_5": _rate(rows, "recall_at_5"),
        "recall_at_10": _rate(rows, "recall_at_10"),
        "source_hit_rate": _rate(rows, "source_hit"),
        "citation_support_rate": _rate(rows, "citation_support_proxy"),
        "claim_support_rate": _rate(rows, "claim_support_proxy"),
        "source_tier_correctness": _rate(rows, "source_tier_correct"),
        "unsupported_answer_rate": round(sum(1 for row in rows if row["unsupported_answer_proxy"]) / max(len(rows), 1), 4),
        "p50_retrieval_latency_ms": _percentile(latencies, 50),
        "p95_retrieval_latency_ms": _percentile(latencies, 95),
        "reranker_latency_ms": round(sum(reranker_latencies) / max(len(reranker_latencies), 1), 3),
    }
    return {"summary": summary, "cases": rows}


def _load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _expected_ids(case: dict[str, Any]) -> set[str]:
    return {
        str(value).strip().lower()
        for value in case.get("expected_source_ids", [])
        if str(value).strip()
    }


def _first_expected_rank(rows: list[dict[str, Any]], expected: set[str]) -> int | None:
    if not expected:
        return None
    for rank, row in enumerate(rows, start=1):
        ids = _row_ids(row)
        if ids & expected:
            return rank
    return None


def _row_ids(row: dict[str, Any]) -> set[str]:
    values = {
        row.get("id"),
        row.get("source_id"),
        row.get("source_name"),
        row.get("parent_id"),
        row.get("title"),
    }
    return {str(value).strip().lower() for value in values if value}


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum(1 for row in rows if row.get(key)) / max(len(rows), 1), 4)


def _mean(values) -> float:
    vals = [float(value) for value in values]
    return round(sum(vals) / max(len(vals), 1), 4)


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if percentile == 50:
        return round(float(median(values)), 3)
    index = math.ceil((percentile / 100) * len(values)) - 1
    return round(values[max(0, min(index, len(values) - 1))], 3)


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
