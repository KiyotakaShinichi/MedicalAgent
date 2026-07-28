"""Validated, opt-in synchronization of governed gold records to a shadow index."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol

from backend.services.managed_vector_store import (
    REMOTE_SAFE_NAMESPACES,
    AzureAISearchAdapter,
    VectorRecord,
    VectorStoreError,
    load_managed_vector_config,
    validate_remote_record,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_GOLD_PATH = Path("Data/lakehouse/gold/vector_records.jsonl")
DEFAULT_OUTPUT_PATH = Path("Data/evals/rag/latest_managed_vector_shadow_sync.json")


class Encoder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> Any:
        ...


def build_managed_vector_shadow_sync(
    *,
    root_dir: str | Path = ROOT_DIR,
    gold_path: str | Path = DEFAULT_GOLD_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    environment: Mapping[str, str] | None = None,
    apply: bool = False,
    batch_size: int = 50,
    encoder: Encoder | None = None,
    transport: Any = None,
) -> dict[str, Any]:
    if batch_size <= 0 or batch_size > 500:
        raise ValueError("batch_size must be between 1 and 500")
    root = Path(root_dir)
    gold_file = _resolve(root, gold_path)
    rows = _read_jsonl(gold_file)
    config = load_managed_vector_config(environment)
    configured = config.provider == "azure_ai_search" and config.configured
    network_allowed = bool(configured and config.allow_network)

    validation_issues: list[dict[str, str]] = []
    for row in rows:
        record_id = str(row.get("record_id") or "")
        try:
            _validate_gold_row(
                row,
                dimension=config.embedding_dimension,
            )
        except (VectorStoreError, ValueError) as exc:
            validation_issues.append(
                {"record_id": record_id, "issue": str(exc)[:240]}
            )

    receipts: list[dict[str, Any]] = []
    indexed_count = 0
    failed_count = 0
    network_performed = False
    if apply:
        if validation_issues:
            raise VectorStoreError("Gold records failed remote-safety validation.")
        if not network_allowed:
            raise VectorStoreError(
                "Managed sync requires Azure AI Search, explicit shadow enablement, "
                "credentials, and NLCARE_MANAGED_VECTOR_ALLOW_NETWORK=true."
            )
        active_encoder = encoder or _load_encoder()
        adapter = AzureAISearchAdapter(config, transport=transport)
        for batch_number, batch in enumerate(_batches(rows, batch_size), start=1):
            records = _embed_batch(
                batch,
                encoder=active_encoder,
                expected_dimension=config.embedding_dimension,
            )
            receipt = _summarize_receipt(
                adapter.upsert(records),
                batch_number=batch_number,
                expected_count=len(records),
            )
            receipts.append(receipt)
            indexed_count += receipt["succeeded"]
            failed_count += receipt["failed"]
            network_performed = True

    validation_passed = bool(rows) and not validation_issues
    sync_completed = bool(apply and network_performed and failed_count == 0)
    status = (
        "shadow_sync_completed"
        if sync_completed
        else "ready_for_opt_in_shadow_sync"
        if validation_passed
        else "needs_attention"
    )
    payload = {
        "schema_version": "nlcare_managed_vector_shadow_sync_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "patient_data_allowed": False,
        "data_scope": "curated_non_patient_kb",
        "live_patient_route_changed": False,
        "provider": "azure_ai_search",
        "gold_path": Path(gold_path).as_posix(),
        "gold_sha256": hashlib.sha256(gold_file.read_bytes()).hexdigest(),
        "record_count": len(rows),
        "batch_size": batch_size,
        "planned_batch_count": math.ceil(len(rows) / batch_size) if rows else 0,
        "configured": configured,
        "network_allowed": network_allowed,
        "network_request_performed": network_performed,
        "validation_passed": validation_passed,
        "validation_issue_count": len(validation_issues),
        "validation_issues": validation_issues,
        "apply_requested": bool(apply),
        "sync_completed": sync_completed,
        "indexed_count": indexed_count,
        "failed_count": failed_count,
        "receipts": receipts,
        "delete_propagation_completed": False,
        "retrieval_improvement_proven": False,
        "promotion_allowed": False,
        "claim_boundary": (
            "This validates or synchronizes curated non-patient engineering records to an isolated "
            "shadow index. It does not change the live route or prove retrieval improvement, clinical "
            "validation, patient benefit, security certification, or production healthcare readiness."
        ),
    }
    destination = _resolve(root, output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _validate_gold_row(
    row: Mapping[str, Any],
    *,
    dimension: int,
) -> None:
    if row.get("data_contract") != "vector_record_gold_v1":
        raise ValueError("Unsupported gold record contract.")
    source_namespace = str(row.get("namespace") or "")
    if source_namespace not in REMOTE_SAFE_NAMESPACES:
        raise ValueError("Gold record namespace is not approved for remote shadow use.")
    validate_remote_record(
        VectorRecord(
            record_id=str(row.get("record_id") or ""),
            vector=tuple(0.0 for _ in range(dimension)),
            text=str(row.get("embedding_input") or ""),
            metadata=dict(row.get("metadata") or {}),
        ),
        namespace=source_namespace,
    )


def _load_encoder() -> Encoder:
    from backend.services.rag_vector_index import _get_encoder

    encoder = _get_encoder()
    if encoder is None:
        raise VectorStoreError(
            "Managed sync requires sentence-transformers/all-MiniLM-L6-v2."
        )
    return encoder


def _embed_batch(
    rows: list[dict[str, Any]],
    *,
    encoder: Encoder,
    expected_dimension: int,
) -> list[VectorRecord]:
    texts = [str(row.get("embedding_input") or "") for row in rows]
    vectors = encoder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    records: list[VectorRecord] = []
    for row, raw in zip(rows, vectors, strict=True):
        vector = tuple(float(value) for value in raw)
        if len(vector) != expected_dimension:
            raise VectorStoreError(
                f"Embedding dimension mismatch: expected {expected_dimension}, got {len(vector)}"
            )
        records.append(
            VectorRecord(
                record_id=str(row["record_id"]),
                vector=vector,
                text=str(row["embedding_input"]),
                metadata=dict(row["metadata"]),
            )
        )
    return records


def _summarize_receipt(
    response: Mapping[str, Any],
    *,
    batch_number: int,
    expected_count: int,
) -> dict[str, Any]:
    rows = response.get("value")
    if not isinstance(rows, list):
        return {
            "batch_number": batch_number,
            "expected": expected_count,
            "succeeded": 0,
            "failed": expected_count,
            "response_rows_present": False,
            "error_codes": ["missing_index_receipts"],
        }
    succeeded = sum(
        bool(row.get("status")) for row in rows if isinstance(row, Mapping)
    )
    failed_rows = [
        row for row in rows if isinstance(row, Mapping) and not bool(row.get("status"))
    ]
    missing = max(expected_count - len(rows), 0)
    return {
        "batch_number": batch_number,
        "expected": expected_count,
        "succeeded": succeeded,
        "failed": len(failed_rows) + missing,
        "response_rows_present": True,
        "error_codes": sorted(
            {
                str(row.get("statusCode") or "indexing_failed")
                for row in failed_rows
            }
        ),
    }


def _batches(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [
        rows[index : index + size]
        for index in range(0, len(rows), size)
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


__all__ = [
    "DEFAULT_GOLD_PATH",
    "DEFAULT_OUTPUT_PATH",
    "ROOT_DIR",
    "build_managed_vector_shadow_sync",
]
