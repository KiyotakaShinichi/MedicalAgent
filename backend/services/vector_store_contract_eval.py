"""Offline contract evaluation for local and managed vector-store adapters."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from backend.services.managed_vector_store import (
    AzureAISearchAdapter,
    InMemoryVectorStore,
    ManagedVectorConfig,
    PineconeAdapter,
    VectorRecord,
    VectorSearchRequest,
    VectorStoreError,
    validate_remote_record,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_GOLD_PATH = Path("Data/lakehouse/gold/vector_records.jsonl")
DEFAULT_OUTPUT_PATH = Path("Data/evals/rag/latest_vector_store_contract_eval.json")


def build_vector_store_contract_eval(
    *,
    root_dir: str | Path = ROOT_DIR,
    gold_path: str | Path = DEFAULT_GOLD_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    root = Path(root_dir)
    records = _read_jsonl(_resolve(root, gold_path))
    cases: list[dict[str, Any]] = []

    fixture_records = [
        VectorRecord(
            record_id="education-a",
            vector=(1.0, 0.0, 0.0, 0.0),
            text="source-backed education fixture",
            metadata=_metadata("education-a", tier="T2", allowed_use=["education"]),
        ),
        VectorRecord(
            record_id="portal-b",
            vector=(0.0, 1.0, 0.0, 0.0),
            text="portal help fixture",
            metadata=_metadata("portal-b", tier="T4", allowed_use=["portal_help"]),
        ),
    ]
    request = VectorSearchRequest(
        query_vector=(1.0, 0.0, 0.0, 0.0),
        text_query="education",
        top_k=2,
    )

    local = InMemoryVectorStore(dimension=4)
    local.upsert(fixture_records)
    local_results = local.search(request)
    cases.append(
        _case(
            "local_contract_ranking",
            bool(local_results and local_results[0].record_id == "education-a"),
            "In-memory contract preserves deterministic vector ranking and tier/use filters.",
        )
    )

    azure_config = _config("azure_ai_search")
    azure = AzureAISearchAdapter(azure_config, transport=_unexpected_transport)
    azure_payload = azure.build_search_payload(request)
    cases.append(
        _case(
            "azure_prefilter_policy",
            azure_payload.get("vectorFilterMode") == "preFilter"
            and "clinical_validation eq false" in str(azure_payload.get("filter"))
            and "patient_facing eq true" in str(azure_payload.get("filter")),
            "Azure AI Search hybrid request applies governance before vector selection.",
        )
    )

    pinecone_config = _config("pinecone")
    pinecone = PineconeAdapter(pinecone_config, transport=_unexpected_transport)
    pinecone_payload = pinecone.build_search_payload(request)
    filter_text = json.dumps(pinecone_payload.get("filter"), sort_keys=True)
    cases.append(
        _case(
            "pinecone_metadata_policy",
            '"clinical_validation"' in filter_text
            and '"patient_facing"' in filter_text
            and '"source_tier"' in filter_text,
            "Pinecone shadow query carries the same canonical governance fields.",
        )
    )

    for name, adapter in (("azure", azure), ("pinecone", pinecone)):
        blocked = False
        try:
            adapter.search(request)
        except VectorStoreError:
            blocked = True
        cases.append(
            _case(
                f"{name}_network_default_off",
                blocked,
                "Managed-vector network execution is blocked without explicit configuration.",
            )
        )

    candidate_failures: list[dict[str, Any]] = []
    for row in records:
        metadata = row.get("metadata") or {}
        vector_record = VectorRecord(
            record_id=str(row.get("record_id") or ""),
            vector=(1.0, 0.0, 0.0, 0.0),
            text=str(row.get("embedding_input") or ""),
            metadata=metadata,
        )
        try:
            validate_remote_record(vector_record, namespace=str(row.get("namespace") or ""))
        except VectorStoreError as exc:
            candidate_failures.append(
                {"record_id": row.get("record_id"), "reason": str(exc)}
            )
    cases.append(
        _case(
            "gold_records_remote_safe",
            bool(records) and not candidate_failures,
            "All generated vector records satisfy the provider-neutral banned-identity-metadata contract.",
        )
    )

    pass_count = sum(case["passed"] for case in cases)
    status = "strong_contract_only" if pass_count == len(cases) else "needs_attention"
    payload = {
        "schema_version": "nlcare_vector_store_contract_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "live_patient_route_backend": "local_faiss_bm25_rrf",
        "managed_backend_mode": "shadow_only_disabled_by_default",
        "managed_network_request_performed": False,
        "managed_vector_comparison_completed": False,
        "retrieval_improvement_proven": False,
        "gold_record_count": len(records),
        "gold_record_validation_failures": candidate_failures,
        "n_cases": len(cases),
        "passed": pass_count,
        "failed": len(cases) - pass_count,
        "pass_rate": round(pass_count / len(cases), 4) if cases else 0.0,
        "cases": cases,
        "provider_matrix": {
            "local_faiss": {
                "role": "live_default_and_fallback",
                "hybrid": True,
                "network": False,
                "current_quality_evidence": "existing frozen baseline comparison",
            },
            "azure_ai_search": {
                "role": "managed_hybrid_shadow_candidate",
                "hybrid": True,
                "pre_filter_governance": True,
                "configured": False,
            },
            "pinecone": {
                "role": "managed_vector_shadow_candidate",
                "hybrid": "client_side_sparse_dense_comparison_required",
                "metadata_filter_governance": True,
                "configured": False,
            },
        },
        "promotion_requirements": [
            "Run the same frozen goldset through local and managed candidates.",
            "Preserve source-tier correctness and refusal correctness.",
            "Do not regress citation precision or unsupported-context rate.",
            "Report p50/p95 latency, ingestion freshness, and measured cost.",
            "Demonstrate delete/rebuild and local fallback recovery.",
            "Keep PHI and patient-specific memory out of managed indexes.",
        ],
        "claim_boundary": (
            "This verifies adapter schemas and metadata-policy behavior offline. It is not a live "
            "Azure or Pinecone benchmark, does not prove retrieval improvement, and does not establish "
            "clinical validation or production healthcare readiness."
        ),
    }
    destination = _resolve(root, output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _metadata(record_id: str, *, tier: str, allowed_use: list[str]) -> dict[str, Any]:
    return {
        "source_id": record_id,
        "chunk_id": record_id,
        "parent_id": record_id,
        "source_tier": tier,
        "allowed_use": allowed_use,
        "patient_facing": True,
        "staleness_status": "current",
        "kb_fingerprint": "fixture-kb",
        "doc_type": "knowledge_chunk",
        "data_scope": "curated_non_patient_kb",
        "clinical_validation": False,
    }


def _config(provider: str) -> ManagedVectorConfig:
    return ManagedVectorConfig(
        provider=provider,
        enabled=False,
        shadow_only=True,
        allow_network=False,
        namespace="nlcare_kb_demo_t1_t3",
        endpoint="https://disabled.example",
        index_name="nlcare-shadow",
        api_version="2026-04-01" if provider == "azure_ai_search" else "2025-10",
        credential="not-used",
        embedding_dimension=4,
    )


def _unexpected_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout: float,
) -> Mapping[str, Any]:
    raise AssertionError("Offline contract evaluation must not perform network requests.")


def _case(case_id: str, passed: bool, description: str) -> dict[str, Any]:
    return {"case_id": case_id, "passed": bool(passed), "description": description}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


__all__ = [
    "DEFAULT_GOLD_PATH",
    "DEFAULT_OUTPUT_PATH",
    "ROOT_DIR",
    "build_vector_store_contract_eval",
]
