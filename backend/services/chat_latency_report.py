from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from backend.database import SessionLocal
from backend.models import RAGEvaluationLog


DEFAULT_OUTPUT_PATH = "Data/evals/latency/latest_chat_latency_report.json"


def build_chat_latency_report(db=None, output_path: str | None = DEFAULT_OUTPUT_PATH, limit: int = 500) -> dict:
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
    finally:
        if owns_session:
            db.close()

    latencies = [float(row.latency_ms) for row in rows if row.latency_ms is not None]
    by_intent = _group_latency(rows, "intent")
    by_cache = _group_latency(rows, "cache_status")
    slowest = [
        {
            "id": row.id,
            "intent": row.intent,
            "cache_status": row.cache_status,
            "latency_ms": row.latency_ms,
            "terminal_step": row.terminal_step,
            "query_preview": row.query_preview,
            "created_at": str(row.created_at),
        }
        for row in sorted(rows, key=lambda item: item.latency_ms or 0, reverse=True)[:10]
        if row.latency_ms is not None
    ]

    report = {
        "schema_version": "chat_latency_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "available" if latencies else "unavailable",
        "sample_size": len(latencies),
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
        "mean_latency_ms": round(float(np.mean(latencies)), 1) if latencies else None,
        "by_intent": by_intent,
        "by_cache_status": by_cache,
        "slowest_traces": slowest,
        "recommendations": _recommendations(latencies, by_intent, by_cache),
        "interpretation": (
            "Use this to tune patient-chat responsiveness. Casual/tool routes should be fast; "
            "RAG routes can be slower if the UI streams stage updates."
        ),
    }
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _group_latency(rows, attr: str) -> dict:
    grouped = {}
    for row in rows:
        value = getattr(row, attr, None) or "unknown"
        if row.latency_ms is None:
            continue
        grouped.setdefault(value, []).append(float(row.latency_ms))
    return {
        key: {
            "count": len(values),
            "p50_ms": _percentile(values, 50),
            "p95_ms": _percentile(values, 95),
            "mean_ms": round(float(np.mean(values)), 1),
        }
        for key, values in sorted(grouped.items())
    }


def _percentile(values: list[float], pct: int) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(values, pct)), 1)


def _recommendations(latencies: list[float], by_intent: dict, by_cache: dict) -> list[str]:
    if not latencies:
        return [
            "No agent trace latency rows yet. Run several support-agent conversations, then regenerate this report.",
        ]
    recommendations = []
    if (_percentile(latencies, 95) or 0) > 5000:
        recommendations.append("p95 latency is high; keep SSE stage updates visible and prefer cached RAG when safe.")
    rag = by_intent.get("education") or by_intent.get("rag") or {}
    if rag and (rag.get("p50_ms") or 0) > 2500:
        recommendations.append("RAG route is the slow path; prebuild FAISS index and defer reranker unless top scores are close.")
    hit = by_cache.get("hit") or by_cache.get("semantic_hit") or {}
    miss = by_cache.get("miss") or {}
    if hit and miss and (hit.get("mean_ms") or 0) >= (miss.get("mean_ms") or 0):
        recommendations.append("Cache hits are not faster than misses; inspect cache lookup and serialization overhead.")
    if not recommendations:
        recommendations.append("Latency profile is demo-friendly; continue logging p50/p95 by route after UI streaming polish.")
    return recommendations
