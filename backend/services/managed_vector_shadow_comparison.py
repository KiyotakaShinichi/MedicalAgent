"""Frozen local-versus-managed vector shadow comparison.

The default execution emits a readiness artifact and performs no network call.
When an Azure AI Search shadow index is explicitly configured, the same frozen
internal retrieval goldset can be queried without changing the live patient
route. This is engineering comparison evidence only.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from backend.services.managed_vector_store import (
    AzureAISearchAdapter,
    VectorSearchRequest,
    VectorStoreError,
    load_managed_vector_config,
)
from backend.services.rag_baseline_comparison import _score_case


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_GOLDSET_PATH = Path("Data/evals/rag/retrieval_goldset.jsonl")
DEFAULT_LOCAL_BASELINE_PATH = Path("Data/evals/rag/latest_rag_baseline_comparison.json")
DEFAULT_OUTPUT_PATH = Path("Data/evals/rag/latest_managed_vector_shadow_comparison.json")
DEFAULT_FAILURE_PATH = Path("Data/evals/rag/latest_managed_vector_shadow_failures.json")
FULL_STACK_ID = "hybrid_rrf_query_rewrite_parent_child_source_tier"


def build_managed_vector_shadow_comparison(
    *,
    root_dir: str | Path = ROOT_DIR,
    goldset_path: str | Path = DEFAULT_GOLDSET_PATH,
    local_baseline_path: str | Path = DEFAULT_LOCAL_BASELINE_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    failure_path: str | Path = DEFAULT_FAILURE_PATH,
    environment: Mapping[str, str] | None = None,
    managed_case_results: Mapping[str, list[dict[str, Any]]] | None = None,
    managed_case_latencies_ms: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    root = Path(root_dir)
    goldset_file = _resolve(root, goldset_path)
    cases = _read_jsonl(goldset_file)
    local_payload = _read_json(_resolve(root, local_baseline_path))
    local_configurations = local_payload.get("configurations") or {}
    local_full = (local_configurations.get(FULL_STACK_ID) or {}).get("summary") or {}
    local_bm25 = (local_configurations.get("bm25_only") or {}).get("summary") or {}
    config = load_managed_vector_config(environment)
    configured = config.provider == "azure_ai_search" and config.configured
    network_allowed = bool(configured and config.allow_network)

    result_map = managed_case_results
    latency_map = dict(managed_case_latencies_ms or {})
    network_performed = False
    readiness_reason = ""
    if result_map is None and network_allowed:
        result_map, latency_map = _run_live_azure_shadow(cases, environment=environment)
        network_performed = True
    elif result_map is None:
        result_map = {}
        readiness_reason = (
            "Azure AI Search shadow endpoint is not explicitly configured for network execution."
        )

    rows: list[dict[str, Any]] = []
    missing_case_ids: list[str] = []
    for case in cases:
        case_id = str(case.get("case_id") or "")
        if case_id not in result_map:
            missing_case_ids.append(case_id)
            continue
        ranked = [_normalize_managed_row(row) for row in result_map[case_id]]
        rows.append(
            _score_case(
                "azure_ai_search_shadow",
                case,
                ranked,
                float(latency_map.get(case_id, 0.0)),
            )
        )

    completed = bool(cases) and not missing_case_ids and len(rows) == len(cases)
    managed_summary = _summary(rows) if completed else None
    joint_improvement = _joint_improvement(managed_summary, local_full) if completed else False
    failures = [row for row in rows if row.get("failure_reasons")]
    status = (
        "acceptable_shadow_comparison"
        if completed
        else "ready_for_managed_shadow_run"
        if cases and local_full
        else "needs_attention"
    )
    candidate_decision = "HOLD"
    if completed and joint_improvement:
        candidate_decision = "REVIEW_FOR_SHADOW_EXTENSION"

    payload = {
        "schema_version": "nlcare_managed_vector_shadow_comparison_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "patient_data_allowed": False,
        "live_patient_route_changed": False,
        "provider": "azure_ai_search",
        "comparison_completed": completed,
        "configured": configured,
        "network_allowed": network_allowed,
        "managed_network_request_performed": network_performed,
        "readiness_reason": readiness_reason,
        "goldset": {
            "path": Path(goldset_path).as_posix(),
            "sha256": hashlib.sha256(goldset_file.read_bytes()).hexdigest(),
            "case_count": len(cases),
            "was_used_for_tuning": False,
            "authorship": "internal_frozen",
        },
        "missing_case_ids": missing_case_ids,
        "local_baselines": {
            "bm25_only": local_bm25,
            "source_governed_full_stack": local_full,
        },
        "managed_summary": managed_summary,
        "comparison": _comparison(managed_summary, local_full, local_bm25),
        "operational_evidence": {
            "measured_cost_usd": None,
            "cost_measurement_completed": False,
            "ingestion_freshness_measured": False,
            "delete_propagation_drill_completed": False,
            "managed_outage_fallback_drill_completed": False,
            "local_fallback_remains_canonical": True,
        },
        "quality_governance_joint_improvement_proven": joint_improvement,
        "retrieval_improvement_proven": joint_improvement,
        "candidate_decision": candidate_decision,
        "failure_count": len(failures),
        "case_rows": rows,
        "claim_boundary": (
            "This is a frozen internal engineering shadow comparison. It is not external validation, "
            "clinical validation, a real-world safety guarantee, or evidence of production healthcare "
            "readiness. The local FAISS/BM25 route remains canonical unless joint quality, governance, "
            "latency, cost, deletion, freshness, and recovery evidence is complete."
        ),
    }
    failure_payload = {
        "schema_version": "nlcare_managed_vector_shadow_failures_v1",
        "generated_at": payload["generated_at"],
        "status": "available" if completed else "not_run",
        "clinical_validation": False,
        "comparison_completed": completed,
        "failure_count": len(failures),
        "failures": failures,
        "claim_boundary": payload["claim_boundary"],
    }
    _write_json(_resolve(root, output_path), payload)
    _write_json(_resolve(root, failure_path), failure_payload)
    return payload


def _run_live_azure_shadow(
    cases: list[dict[str, Any]],
    *,
    environment: Mapping[str, str] | None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
    config = load_managed_vector_config(environment)
    if config.provider != "azure_ai_search" or not config.configured or not config.allow_network:
        raise VectorStoreError("Azure AI Search network gates are incomplete.")
    try:
        from backend.services.rag_vector_index import _get_encoder
    except ImportError as exc:
        raise VectorStoreError("The local sentence-transformer encoder is unavailable.") from exc
    encoder = _get_encoder()
    if encoder is None:
        raise VectorStoreError(
            "Live managed comparison requires sentence-transformers/all-MiniLM-L6-v2."
        )

    adapter = AzureAISearchAdapter(config)
    results: dict[str, list[dict[str, Any]]] = {}
    latencies: dict[str, float] = {}
    for case in cases:
        case_id = str(case.get("case_id") or "")
        query = str(case.get("user_query") or case.get("query") or "")
        raw = encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        started = time.perf_counter()
        response = adapter.search(
            VectorSearchRequest(
                query_vector=tuple(float(value) for value in raw),
                text_query=query,
                top_k=10,
                allowed_tiers=tuple(case.get("acceptable_source_tiers") or ("T1", "T2", "T3")),
            )
        )
        latencies[case_id] = (time.perf_counter() - started) * 1000.0
        results[case_id] = [
            {
                "record_id": item.record_id,
                "score": item.score,
                "text": item.text,
                "metadata": dict(item.metadata),
            }
            for item in response
        ]
    return results, latencies


def _normalize_managed_row(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    return {
        **metadata,
        "id": row.get("record_id") or row.get("id"),
        "chunk_id": metadata.get("chunk_id") or row.get("record_id") or row.get("id"),
        "parent_id": metadata.get("parent_id") or metadata.get("source_id"),
        "source_id": metadata.get("source_id"),
        "tier": metadata.get("source_tier"),
        "text": row.get("text") or row.get("content") or "",
        "retrieval_score": float(row.get("score") or row.get("@search.score") or 0.0),
        "vector_score": float(row.get("score") or row.get("@search.score") or 0.0),
        "retrieval_backend": "azure_ai_search_shadow",
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row.get("latency_ms") or 0.0) for row in rows]
    return {
        "case_count": len(rows),
        "recall_at_5": _mean(row["recall_at_5"] for row in rows),
        "recall_at_10": _mean(row["recall_at_10"] for row in rows),
        "mrr": _mean(row["mrr"] for row in rows),
        "ndcg_at_10": _mean(row["ndcg_at_10"] for row in rows),
        "citation_precision": _mean(row["citation_precision"] for row in rows),
        "claim_support_rate": _rate(row["claim_supported"] for row in rows),
        "unsupported_context_rate": _rate(row["unsupported_context"] for row in rows),
        "refusal_correctness": _rate(row["refusal_correct"] for row in rows),
        "source_tier_correctness": _rate(row["source_tier_correct"] for row in rows),
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
    }


def _comparison(
    managed: Mapping[str, Any] | None,
    local_full: Mapping[str, Any],
    local_bm25: Mapping[str, Any],
) -> dict[str, Any]:
    if not managed:
        return {
            "available": False,
            "managed_vs_full_stack": None,
            "managed_vs_bm25": None,
        }
    metrics = (
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "ndcg_at_10",
        "citation_precision",
        "claim_support_rate",
        "unsupported_context_rate",
        "refusal_correctness",
        "source_tier_correctness",
        "latency_p50_ms",
        "latency_p95_ms",
    )
    return {
        "available": True,
        "managed_vs_full_stack": {
            key: round(float(managed.get(key, 0)) - float(local_full.get(key, 0)), 4)
            for key in metrics
        },
        "managed_vs_bm25": {
            key: round(float(managed.get(key, 0)) - float(local_bm25.get(key, 0)), 4)
            for key in metrics
        },
    }


def _joint_improvement(
    managed: Mapping[str, Any] | None,
    local_full: Mapping[str, Any],
) -> bool:
    if not managed or not local_full:
        return False
    return bool(
        float(managed.get("recall_at_10", 0)) > float(local_full.get("recall_at_10", 0))
        and float(managed.get("citation_precision", 0))
        >= float(local_full.get("citation_precision", 0))
        and float(managed.get("unsupported_context_rate", 1))
        <= float(local_full.get("unsupported_context_rate", 1))
        and float(managed.get("source_tier_correctness", 0)) >= 1.0
        and float(managed.get("refusal_correctness", 0)) >= 1.0
    )


def _mean(values: Any) -> float:
    rows = [float(value) for value in values]
    return round(statistics.fmean(rows), 4) if rows else 0.0


def _rate(values: Any) -> float:
    rows = [bool(value) for value in values]
    return round(sum(rows) / len(rows), 4) if rows else 0.0


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction, 3)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


__all__ = [
    "DEFAULT_FAILURE_PATH",
    "DEFAULT_GOLDSET_PATH",
    "DEFAULT_LOCAL_BASELINE_PATH",
    "DEFAULT_OUTPUT_PATH",
    "FULL_STACK_ID",
    "ROOT_DIR",
    "build_managed_vector_shadow_comparison",
]
