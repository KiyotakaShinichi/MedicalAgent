from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.agent_query_rewriting import rewrite_and_decompose
from backend.services.agent_retrieval import (
    contextual_compression,
    expand_parent_child_windows,
    hybrid_retrieval,
    rerank_context,
)
from backend.services.cross_encoder_reranker import cross_encoder_available, cross_encoder_feature_enabled


OUTPUT_PATH = Path("Data/evals/rag/latest_reranker_ablation.json")


CASES = [
    {"case_id": "cbc_low_wbc", "query": "What does low WBC mean during chemo?", "intent": "education", "expected_terms": ["wbc", "cbc"]},
    {"case_id": "vus_general", "query": "What does a VUS mean in genetic testing?", "intent": "education", "expected_terms": ["vus", "genetic"]},
    {"case_id": "tumor_marker_limits", "query": "Does CA 15-3 prove recurrence?", "intent": "education", "expected_terms": ["ca 15-3", "tumor marker"]},
    {"case_id": "portal_upload", "query": "How do I upload an MRI report?", "intent": "portal_help", "expected_terms": ["upload", "mri"]},
    {"case_id": "fever_after_chemo", "query": "Fever after chemo, what should I do?", "intent": "safety_boundary", "expected_terms": ["fever", "urgent"]},
]


def main() -> None:
    baseline = _run(enabled=False)
    cross_encoder = _run(enabled=True)
    payload = {
        "schema_version": "reranker_ablation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "acceptable" if cross_encoder["summary"]["unsupported_answer_rate"] == 0 else "needs_attention",
        "cross_encoder_available": cross_encoder_available(),
        "cross_encoder_enabled_for_eval": cross_encoder_feature_enabled(),
        "summary": {
            "before_live_rag_pass_rate_proxy": baseline["summary"]["retrieval_hit_rate"],
            "after_live_rag_pass_rate_proxy": cross_encoder["summary"]["retrieval_hit_rate"],
            "before_unsupported_answer_rate": baseline["summary"]["unsupported_answer_rate"],
            "after_unsupported_answer_rate": cross_encoder["summary"]["unsupported_answer_rate"],
            "before_source_tier_correctness": baseline["summary"]["source_tier_correctness"],
            "after_source_tier_correctness": cross_encoder["summary"]["source_tier_correctness"],
            "p50_retrieval_latency_ms": cross_encoder["summary"]["p50_retrieval_latency_ms"],
            "p95_retrieval_latency_ms": cross_encoder["summary"]["p95_retrieval_latency_ms"],
            "reranker_latency_ms": cross_encoder["summary"]["reranker_latency_ms"],
        },
        "baseline_disabled": baseline,
        "cross_encoder_or_fallback": cross_encoder,
        "claim_boundary": (
            "Reranker ablation is retrieval engineering evidence only. A cross-encoder "
            "can improve precision but does not guarantee medical correctness."
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


def _run(*, enabled: bool) -> dict:
    previous = os.environ.get("RAG_ENABLE_CROSS_ENCODER")
    os.environ["RAG_ENABLE_CROSS_ENCODER"] = "true" if enabled else "false"
    rows = []
    latencies = []
    reranker_latencies = []
    try:
        for case in CASES:
            started = time.perf_counter()
            rewritten = rewrite_and_decompose(case["query"], case["intent"])
            retrieved = hybrid_retrieval(rewritten, case["intent"])
            expanded = expand_parent_child_windows(retrieved)
            reranked = rerank_context(expanded, rewritten, case["intent"], {"level": "routine"})
            compressed = contextual_compression(reranked)
            latency = (time.perf_counter() - started) * 1000
            latencies.append(latency)
            reranker_latencies.extend([
                float(item.get("cross_encoder_latency_ms") or 0.0)
                for item in compressed
                if item.get("cross_encoder_latency_ms") is not None
            ])
            joined = " ".join((item.get("title") or "") + " " + (item.get("text") or "") for item in compressed).lower()
            hit = any(term.lower() in joined for term in case["expected_terms"])
            source_tier_ok = all((item.get("source_tier") or "T1") != "T5" for item in compressed)
            rows.append({
                "case_id": case["case_id"],
                "retrieval_hit": hit,
                "source_hit": bool(compressed),
                "citation_support_proxy": hit,
                "claim_support_proxy": hit,
                "source_tier_correct": source_tier_ok,
                "unsupported_answer_proxy": not hit,
                "backend": [item.get("reranker_backend") for item in compressed],
                "latency_ms": round(latency, 3),
            })
    finally:
        if previous is None:
            os.environ.pop("RAG_ENABLE_CROSS_ENCODER", None)
        else:
            os.environ["RAG_ENABLE_CROSS_ENCODER"] = previous
    return {
        "summary": {
            "case_count": len(rows),
            "retrieval_hit_rate": _rate(rows, "retrieval_hit"),
            "source_hit_rate": _rate(rows, "source_hit"),
            "citation_support_rate": _rate(rows, "citation_support_proxy"),
            "claim_support_rate": _rate(rows, "claim_support_proxy"),
            "source_tier_correctness": _rate(rows, "source_tier_correct"),
            "unsupported_answer_rate": round(sum(1 for row in rows if row["unsupported_answer_proxy"]) / max(len(rows), 1), 4),
            "p50_retrieval_latency_ms": _percentile(latencies, 50),
            "p95_retrieval_latency_ms": _percentile(latencies, 95),
            "reranker_latency_ms": round(sum(reranker_latencies) / max(len(reranker_latencies), 1), 3),
        },
        "cases": rows,
    }


def _rate(rows: list[dict], key: str) -> float:
    return round(sum(1 for row in rows if row.get(key)) / max(len(rows), 1), 4)


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((percentile / 100) * (len(values) - 1))))
    return round(values[index], 3)


if __name__ == "__main__":
    main()
