from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_BASELINE_PATH = "Data/evals/rag/latest_rag_baseline_comparison.json"
DEFAULT_OUTPUT_PATH = "Data/evals/rag/latest_pinecone_shadow_retrieval_comparison.json"
DEFAULT_DOC_PATH = "docs/pinecone_shadow_retrieval.md"

FULL_STACK_ID = "hybrid_rrf_query_rewrite_parent_child_source_tier"

CLAIM_BOUNDARY = (
    "Pinecone shadow retrieval is an optional managed-vector-search comparison scaffold. It is disabled by "
    "default, must not store PHI or raw patient chat, and does not replace source-tier filtering, allowed-use "
    "filtering, citation validation, safety refusal, or local FAISS/BM25 fallback. This artifact is not clinical "
    "validation, not healthcare production readiness, and not proof of retrieval improvement until a real shadow "
    "run is completed on frozen evals."
)


def build_pinecone_shadow_retrieval_comparison(
    *,
    baseline_path: str | Path = DEFAULT_BASELINE_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    doc_path: str | Path = DEFAULT_DOC_PATH,
    allow_network: bool = False,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    env_values = env or dict(os.environ)
    config = _pinecone_config(env_values)
    baseline = _read_json(_resolve(baseline_path))
    local_metrics = _local_reference_metrics(baseline)
    status = _status(configured=config["configured"], allow_network=allow_network)

    payload: dict[str, Any] = {
        "schema_version": "pinecone_shadow_retrieval_comparison_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "live_patient_route_enabled": False,
        "phi_allowed": False,
        "pinecone_config": config,
        "comparison_completed": False,
        "network_execution_allowed": bool(allow_network),
        "local_reference_metrics": local_metrics,
        "pinecone_metrics": None,
        "delta_vs_local": None,
        "namespace_contract": {
            "nlcare_kb_demo_t1_t3": "synthetic/demo patient-facing KB chunks only",
            "nlcare_eval_synthetic": "frozen eval chunks and synthetic fixtures only",
            "nlcare_clinician_only_shadow": "disabled by default; never cited to patient-facing routes",
            "patient_data": "disallowed until compliance/security review",
        },
        "metadata_filter_contract": [
            "source_tier",
            "allowed_use",
            "patient_facing",
            "staleness_status",
            "kb_fingerprint",
            "clinical_validation=false",
            "doc_type",
        ],
        "upsert_record_schema": {
            "id": "chunk_id or source_id:section",
            "values": "embedding vector from the approved embedding model",
            "metadata": {
                "source_id": "stable source identifier",
                "chunk_id": "stable chunk identifier",
                "source_tier": "T1/T2/T3/T4/T5",
                "allowed_use": "patient_education / clinician_only / safety_policy / etc.",
                "patient_facing": "boolean",
                "staleness_status": "current / stale / unknown",
                "kb_fingerprint": "knowledge-base fingerprint used for invalidation",
                "clinical_validation": False,
            },
        },
        "shadow_eval_plan": {
            "goldset": "Data/evals/rag/retrieval_goldset.jsonl",
            "metrics": [
                "Recall@5",
                "Recall@10",
                "MRR",
                "NDCG@10",
                "citation_precision",
                "claim_support_rate",
                "unsupported_context_rate",
                "source_tier_correctness",
                "refusal_correctness",
                "latency_p50_ms",
                "latency_p95_ms",
                "estimated_cost_per_1k_queries",
            ],
            "comparison_rule": "dual-run local FAISS/BM25/full-stack and Pinecone under the same frozen cases.",
        },
        "promotion_gate": {
            "pinecone_can_replace_local_retrieval": False,
            "minimum_before_promotion": [
                "no PHI or patient-specific memory in Pinecone",
                "source_tier_correctness remains 1.0",
                "unsafe_answer_rate remains 0.0 in live-agent eval",
                "citation_precision does not regress versus local full stack",
                "unsupported_context_rate does not increase versus local full stack",
                "latency/cost tradeoff is explicitly reported",
                "local FAISS/BM25 fallback remains available",
            ],
        },
        "blocked_claims": [
            "clinical validation",
            "retrieval improvement proven",
            "production healthcare readiness",
            "HIPAA compliance",
            "real patient safety",
            "patient-facing clinical confidence",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(_resolve(output_path), payload)
    _write_doc(_resolve(doc_path), payload)
    return payload


def _status(*, configured: bool, allow_network: bool) -> str:
    if configured and allow_network:
        return "configured_ready_for_manual_shadow_run"
    if configured:
        return "configured_dry_run_only"
    return "ready_for_shadow_mode_not_configured"


def _pinecone_config(env: dict[str, str]) -> dict[str, Any]:
    enabled = _truthy(env.get("PINECONE_ENABLED"))
    api_key_present = bool((env.get("PINECONE_API_KEY") or "").strip())
    index_host_present = bool((env.get("PINECONE_INDEX_HOST") or "").strip())
    namespace = (env.get("PINECONE_NAMESPACE_KB") or "nlcare_kb_demo_t1_t3").strip()
    return {
        "enabled": enabled,
        "configured": enabled and api_key_present and index_host_present,
        "api_key_present": api_key_present,
        "index_host_present": index_host_present,
        "namespace": namespace,
        "required_env": [
            "PINECONE_ENABLED=false by default",
            "PINECONE_API_KEY",
            "PINECONE_INDEX_HOST",
            "PINECONE_NAMESPACE_KB=nlcare_kb_demo_t1_t3",
        ],
        "disabled_reason": None if enabled else "PINECONE_ENABLED is not true",
    }


def _local_reference_metrics(baseline: dict[str, Any]) -> dict[str, Any]:
    configurations = baseline.get("configurations") or {}
    bm25 = (configurations.get("bm25_only") or {}).get("summary") or {}
    full = (configurations.get(FULL_STACK_ID) or {}).get("summary") or {}
    return {
        "baseline_artifact": "Data/evals/rag/latest_rag_baseline_comparison.json",
        "baseline_status": baseline.get("status"),
        "total_n": baseline.get("total_n"),
        "clinical_validation": baseline.get("clinical_validation", False),
        "bm25_only": _metric_subset(bm25),
        "source_governed_full_stack": _metric_subset(full),
        "current_honest_reading": (
            "Use Pinecone only as a shadow comparison. Existing local evidence shows governance value, "
            "but raw recall superiority over BM25 is not proven."
        ),
    }


def _metric_subset(summary: dict[str, Any]) -> dict[str, Any]:
    keys = [
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
    ]
    return {key: summary.get(key) for key in keys}


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    local = payload["local_reference_metrics"]
    full = local["source_governed_full_stack"]
    lines = [
        "# Pinecone Shadow Retrieval Comparison",
        "",
        payload["claim_boundary"],
        "",
        "## Current Status",
        "",
        f"- Status: `{payload['status']}`",
        f"- Pinecone configured: `{payload['pinecone_config']['configured']}`",
        f"- Comparison completed: `{payload['comparison_completed']}`",
        f"- PHI allowed: `{payload['phi_allowed']}`",
        f"- Live patient route enabled: `{payload['live_patient_route_enabled']}`",
        "",
        "## Local Reference",
        "",
        f"- Full-stack Recall@10: `{full.get('recall_at_10')}`",
        f"- Full-stack citation precision: `{full.get('citation_precision')}`",
        f"- Full-stack unsupported context rate: `{full.get('unsupported_context_rate')}`",
        f"- Full-stack source-tier correctness: `{full.get('source_tier_correctness')}`",
        "",
        "## Metadata Filter Contract",
        "",
        *[f"- `{item}`" for item in payload["metadata_filter_contract"]],
        "",
        "## Promotion Gate",
        "",
        *[f"- {item}" for item in payload["promotion_gate"]["minimum_before_promotion"]],
        "",
        "## Blocked Claims",
        "",
        *[f"- {item}" for item in payload["blocked_claims"]],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


__all__ = [
    "CLAIM_BOUNDARY",
    "DEFAULT_BASELINE_PATH",
    "DEFAULT_DOC_PATH",
    "DEFAULT_OUTPUT_PATH",
    "build_pinecone_shadow_retrieval_comparison",
]
