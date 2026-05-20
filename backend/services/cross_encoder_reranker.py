"""Optional local cross-encoder reranking for the RAG retrieval stack.

This module is intentionally dependency-light by default.  If
``RAG_ENABLE_CROSS_ENCODER`` is not enabled, or if ``sentence_transformers`` /
the configured model cannot be loaded, reranking falls back to the existing
heuristic order while preserving every metadata field.  The cross-encoder is
an extra precision layer after dense+sparse RRF; it never bypasses source
governance, citation validation, or safety validators.
"""

from __future__ import annotations

import importlib.util
import os
import time
from functools import lru_cache
from typing import Any, Mapping


DEFAULT_CROSS_ENCODER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_CANDIDATE_LIMIT = 40


def cross_encoder_feature_enabled() -> bool:
    return os.getenv("RAG_ENABLE_CROSS_ENCODER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def cross_encoder_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> tuple[Any | None, str | None]:
    if not cross_encoder_feature_enabled():
        return None, "feature_disabled"
    if not cross_encoder_available():
        return None, "sentence_transformers_unavailable"
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(model_name), None
    except Exception as exc:  # noqa: BLE001 - safe fallback is the contract
        return None, f"model_load_failed:{str(exc)[:160]}"


def rerank_with_cross_encoder(
    query: str,
    candidates: list[Mapping[str, Any]],
    *,
    top_k: int = 5,
    candidate_limit: int | None = None,
    model_name: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return reranked rows plus telemetry.

    The input rows are copied as dictionaries and all original metadata is
    preserved.  A disabled/unavailable model is not an error; callers still
    get the top heuristic rows with ``reranker_backend`` set to a fallback
    label and latency telemetry populated.
    """
    started = time.perf_counter()
    model_name = model_name or os.getenv("RAG_CROSS_ENCODER_MODEL", DEFAULT_CROSS_ENCODER_MODEL)
    limit = candidate_limit or int(os.getenv("RAG_CROSS_ENCODER_CANDIDATE_LIMIT", str(DEFAULT_CANDIDATE_LIMIT)))
    limited = [dict(row) for row in candidates[: max(1, limit)]]
    model, error = _load_model(model_name)

    if model is None or not limited:
        elapsed = (time.perf_counter() - started) * 1000
        backend = "heuristic_fallback"
        if error:
            backend = f"{backend}:{error}"
        rows = []
        for rank, row in enumerate(limited[:top_k], start=1):
            rows.append({
                **row,
                "cross_encoder_score": None,
                "cross_encoder_rank": rank,
                "reranker_backend": row.get("reranker_backend") or backend,
                "cross_encoder_latency_ms": round(elapsed, 3),
            })
        return rows, {
            "enabled": cross_encoder_feature_enabled(),
            "available": model is not None,
            "model": model_name if model is not None else None,
            "fallback_reason": error,
            "candidate_count": len(limited),
            "returned_count": len(rows),
            "reranker_latency_ms": round(elapsed, 3),
        }

    pairs = [
        (
            query,
            " ".join(str(row.get(key) or "") for key in ("title", "section", "topic", "text")),
        )
        for row in limited
    ]
    try:
        raw_scores = model.predict(pairs)
        scores = [float(score) for score in raw_scores]
    except Exception as exc:  # noqa: BLE001
        _load_model.cache_clear()
        elapsed = (time.perf_counter() - started) * 1000
        rows = []
        for rank, row in enumerate(limited[:top_k], start=1):
            rows.append({
                **row,
                "cross_encoder_score": None,
                "cross_encoder_rank": rank,
                "reranker_backend": f"heuristic_fallback:model_predict_failed:{str(exc)[:120]}",
                "cross_encoder_latency_ms": round(elapsed, 3),
            })
        return rows, {
            "enabled": True,
            "available": False,
            "model": model_name,
            "fallback_reason": f"model_predict_failed:{str(exc)[:160]}",
            "candidate_count": len(limited),
            "returned_count": len(rows),
            "reranker_latency_ms": round(elapsed, 3),
        }

    scored = []
    for idx, row in enumerate(limited):
        score = scores[idx] if idx < len(scores) else 0.0
        scored.append({
            **row,
            "cross_encoder_score": round(score, 6),
            "reranker_backend": f"cross_encoder:{model_name}",
        })
    scored.sort(key=lambda row: float(row.get("cross_encoder_score") or 0.0), reverse=True)
    elapsed = (time.perf_counter() - started) * 1000
    output = []
    for rank, row in enumerate(scored[:top_k], start=1):
        output.append({
            **row,
            "cross_encoder_rank": rank,
            "cross_encoder_latency_ms": round(elapsed, 3),
        })
    return output, {
        "enabled": True,
        "available": True,
        "model": model_name,
        "fallback_reason": None,
        "candidate_count": len(limited),
        "returned_count": len(output),
        "reranker_latency_ms": round(elapsed, 3),
    }


__all__ = [
    "DEFAULT_CANDIDATE_LIMIT",
    "DEFAULT_CROSS_ENCODER_MODEL",
    "cross_encoder_available",
    "cross_encoder_feature_enabled",
    "rerank_with_cross_encoder",
]
