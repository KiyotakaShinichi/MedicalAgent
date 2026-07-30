"""Executable reliability drills for the curated non-patient data platform."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from backend.services.data_platform_pipeline import _build_silver_rows
from backend.services.managed_vector_store import (
    InMemoryVectorStore,
    ManagedVectorConfig,
    PineconeAdapter,
    VectorRecord,
    VectorSearchRequest,
    VectorStoreError,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_GOLD_PATH = Path("Data/lakehouse/gold/vector_records.jsonl")
DEFAULT_OUTPUT_PATH = Path("Data/evals/ops/latest_data_platform_reliability_eval.json")


def build_data_platform_reliability_eval(
    *,
    root_dir: str | Path = ROOT_DIR,
    gold_path: str | Path = DEFAULT_GOLD_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    root = Path(root_dir)
    gold_rows = _read_jsonl(_resolve(root, gold_path))
    cases: list[dict[str, Any]] = []

    fixture_chunks = [
        {
            "id": "chunk-a",
            "parent_id": "source-a",
            "text": "curated non-patient education fixture",
            "trust_level": "official",
        },
        {
            "id": "chunk-b",
            "parent_id": "source-b",
            "text": "",
            "trust_level": "official",
        },
    ]
    governance = {
        "source-a": {
            "tier": "T2",
            "allowed_use": ["education"],
            "staleness_status": "current",
        },
        "source-b": {
            "tier": "T2",
            "allowed_use": ["education"],
            "staleness_status": "current",
        },
    }
    silver_first, quarantine_first = _build_silver_rows(fixture_chunks, governance)
    silver_second, quarantine_second = _build_silver_rows(fixture_chunks, governance)
    cases.append(
        _case(
            "idempotent_replay",
            _fingerprint(silver_first) == _fingerprint(silver_second)
            and _fingerprint(quarantine_first) == _fingerprint(quarantine_second),
            "Repeated materialization yields byte-stable logical rows.",
        )
    )
    cases.append(
        _case(
            "partial_failure_quarantine",
            len(silver_first) == 1
            and len(quarantine_first) == 1
            and "missing_text" in quarantine_first[0]["issues"],
            "One malformed row is quarantined without suppressing the valid row.",
        )
    )

    legacy = {
        "record_id": "legacy-a",
        "embedding_input": "legacy curated education fixture",
        "namespace": "nlcare_kb_demo_t1_t3",
        "metadata": {
            "source_id": "source-a",
            "chunk_id": "legacy-a",
            "parent_id": "source-a",
            "source_tier": "T2",
            "allowed_use": ["education"],
            "patient_facing": True,
            "staleness_status": "current",
            "kb_fingerprint": "fixture",
        },
        "data_contract": "vector_record_gold_v0",
    }
    migrated = migrate_gold_record(legacy)
    cases.append(
        _case(
            "schema_evolution_v0_to_v1",
            migrated["data_contract"] == "vector_record_gold_v1"
            and migrated["metadata"]["data_scope"] == "curated_non_patient_kb"
            and migrated["metadata"]["clinical_validation"] is False
            and migrated["metadata"]["doc_type"] == "knowledge_chunk",
            "Legacy non-patient records receive explicit v1 governance defaults.",
        )
    )

    sample_rows = gold_rows[:5]
    tombstone_ids = {str(sample_rows[0]["record_id"])} if sample_rows else set()
    active_rows, deleted = apply_tombstones(sample_rows, tombstone_ids)
    cases.append(
        _case(
            "local_delete_propagation",
            bool(sample_rows)
            and deleted == sorted(tombstone_ids)
            and not tombstone_ids.intersection(
                str(row.get("record_id")) for row in active_rows
            ),
            "A local tombstone removes the target record deterministically.",
        )
    )

    batches_first = build_backfill_batches(
        [str(row.get("record_id")) for row in gold_rows],
        batch_size=50,
    )
    batches_second = build_backfill_batches(
        [str(row.get("record_id")) for row in reversed(gold_rows)],
        batch_size=50,
    )
    cases.append(
        _case(
            "deterministic_backfill_batches",
            bool(gold_rows)
            and batches_first == batches_second
            and sum(len(batch) for batch in batches_first) == len(gold_rows),
            "Backfill partitions are stable regardless of input ordering.",
        )
    )

    scale_multiplier = 100
    scaled_ids = [
        f"{row.get('record_id')}:synthetic-scale-{replica:03d}"
        for replica in range(scale_multiplier)
        for row in gold_rows
    ]
    scaled_first = build_backfill_batches(scaled_ids, batch_size=1000)
    scaled_second = build_backfill_batches(reversed(scaled_ids), batch_size=1000)
    scale_deterministic = (
        bool(gold_rows)
        and scaled_first == scaled_second
        and len({record_id for batch in scaled_first for record_id in batch})
        == len(scaled_ids)
    )
    cases.append(
        _case(
            "partition_scale_replay_100x",
            scale_deterministic,
            "A 100x non-patient identifier replay remains unique and partition-deterministic.",
        )
    )

    remote_blocked, local_recovered = _fallback_drill()
    cases.append(
        _case(
            "managed_failure_local_fallback",
            remote_blocked and local_recovered,
            "A disabled managed adapter fails closed while the local contract remains queryable.",
        )
    )

    passed = sum(row["passed"] for row in cases)
    status = "strong_offline_drill" if passed == len(cases) else "needs_attention"
    payload = {
        "schema_version": "nlcare_data_platform_reliability_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "patient_data_processed": False,
        "external_cloud_write_performed": False,
        "gold_record_count": len(gold_rows),
        "n_cases": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "cases": cases,
        "backfill": {
            "batch_size": 50,
            "batch_count": len(batches_first),
            "deterministic": bool(cases[4]["passed"]),
            "managed_upsert_performed": False,
        },
        "partition_scale_replay": {
            "scale_multiplier": scale_multiplier,
            "base_record_count": len(gold_rows),
            "replayed_record_count": len(scaled_ids),
            "batch_size": 1000,
            "batch_count": len(scaled_first),
            "unique_record_ids": len(
                {record_id for batch in scaled_first for record_id in batch}
            ),
            "partition_fingerprint": _fingerprint(scaled_first),
            "deterministic": scale_deterministic,
            "timing_benchmark_performed": False,
            "managed_cloud_throughput_measured": False,
        },
        "delete_propagation": {
            "local_tombstone_drill_completed": bool(cases[3]["passed"]),
            "managed_index_delete_completed": False,
            "cloud_object_delete_completed": False,
        },
        "recovery": {
            "local_fallback_drill_completed": bool(cases[6]["passed"]),
            "azure_restore_drill_completed": False,
            "postgres_point_in_time_restore_completed": False,
        },
        "claim_boundary": (
            "These are deterministic offline drills over curated non-patient fixtures and local data "
            "products. They do not prove managed-cloud durability, backup restoration, patient-data "
            "safety, clinical validation, or production healthcare readiness."
        ),
    }
    destination = _resolve(root, output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def migrate_gold_record(record: Mapping[str, Any]) -> dict[str, Any]:
    output = json.loads(json.dumps(record))
    metadata = dict(output.get("metadata") or {})
    metadata.setdefault("data_scope", "curated_non_patient_kb")
    metadata.setdefault("clinical_validation", False)
    metadata.setdefault("doc_type", "knowledge_chunk")
    metadata.setdefault("source_name", "Migrated curated source")
    metadata.setdefault("title", "Migrated curated source")
    metadata.setdefault("source_url", "")
    metadata.setdefault("topic", "")
    metadata.setdefault("section", "")
    metadata.setdefault("tags", [])
    output["metadata"] = metadata
    output["data_contract"] = "vector_record_gold_v1"
    return output


def apply_tombstones(
    records: Iterable[Mapping[str, Any]],
    tombstone_ids: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    active: list[dict[str, Any]] = []
    deleted: list[str] = []
    for record in records:
        record_id = str(record.get("record_id") or "")
        if record_id in tombstone_ids:
            deleted.append(record_id)
        else:
            active.append(dict(record))
    return active, sorted(set(deleted))


def build_backfill_batches(record_ids: Iterable[str], *, batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ordered = sorted({str(record_id) for record_id in record_ids if str(record_id)})
    return [ordered[index : index + batch_size] for index in range(0, len(ordered), batch_size)]


def _fallback_drill() -> tuple[bool, bool]:
    config = ManagedVectorConfig(
        provider="pinecone",
        enabled=False,
        shadow_only=True,
        allow_network=False,
        namespace="nlcare_kb_demo_t1_t3",
        endpoint="https://disabled.example",
        index_name="disabled",
        api_version="2025-10",
        credential="not-used",
        embedding_dimension=4,
    )
    request = VectorSearchRequest(
        query_vector=(1.0, 0.0, 0.0, 0.0),
        text_query="education",
        top_k=1,
    )
    remote_blocked = False
    try:
        PineconeAdapter(config).search(request)
    except VectorStoreError:
        remote_blocked = True

    local = InMemoryVectorStore(dimension=4)
    local.upsert(
        [
            VectorRecord(
                record_id="fallback-a",
                vector=(1.0, 0.0, 0.0, 0.0),
                text="curated education fallback",
                metadata={
                    "source_id": "fallback-source",
                    "chunk_id": "fallback-a",
                    "parent_id": "fallback-source",
                    "source_tier": "T2",
                    "allowed_use": ["education"],
                    "patient_facing": True,
                    "staleness_status": "current",
                    "kb_fingerprint": "fixture",
                    "doc_type": "knowledge_chunk",
                    "data_scope": "curated_non_patient_kb",
                    "clinical_validation": False,
                },
            )
        ]
    )
    local_recovered = bool(local.search(request))
    return remote_blocked, local_recovered


def _case(case_id: str, passed: bool, description: str) -> dict[str, Any]:
    return {"case_id": case_id, "passed": bool(passed), "description": description}


def _fingerprint(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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
    "apply_tombstones",
    "build_backfill_batches",
    "build_data_platform_reliability_eval",
    "migrate_gold_record",
]
