"""Incremental non-patient data pipeline for NLCare knowledge assets.

The pipeline materializes a small local medallion layout:

* bronze: content-addressed copies and a source manifest;
* silver: validated, governance-enriched knowledge chunks;
* gold: provider-neutral records for offline embedding/vector shadow loads.

It deliberately excludes patient records, raw chat, and live clinical data.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parents[2]
SUPPORTED_SOURCE_EXTENSIONS = {".md", ".txt", ".pdf", ".json"}
REMOTE_SAFE_NAMESPACES = {
    "nlcare_kb_demo_t1_t3",
    "nlcare_portal_help",
}
BANNED_IDENTITY_METADATA_KEYS = {
    "address",
    "email",
    "mrn",
    "name",
    "patient_id",
    "phone",
    "raw_chat",
    "raw_prompt",
    "raw_response",
}
REQUIRED_GOLD_METADATA_KEYS = {
    "allowed_use",
    "chunk_id",
    "clinical_validation",
    "data_scope",
    "doc_type",
    "kb_fingerprint",
    "parent_id",
    "patient_facing",
    "source_id",
    "source_tier",
    "staleness_status",
}


def run_data_platform_pipeline(
    *,
    root_dir: str | Path = ROOT_DIR,
    source_dir: str | Path = "KnowledgeBase/raw",
    chunk_artifact: str | Path = "Data/rag_knowledge_base_chunks.json",
    governance_artifact: str | Path = "Data/evals/rag/latest_kb_source_governance.json",
    contract_path: str | Path = "config/data_contracts.json",
    output_dir: str | Path = "Data/lakehouse",
) -> dict[str, Any]:
    root = Path(root_dir)
    source_root = _resolve(root, source_dir)
    output_root = _resolve(root, output_dir)
    bronze_root = output_root / "bronze" / "knowledge_sources"
    silver_path = output_root / "silver" / "knowledge_chunks.jsonl"
    gold_path = output_root / "gold" / "vector_records.jsonl"
    quarantine_path = output_root / "quarantine" / "knowledge_chunks.jsonl"
    manifest_path = output_root / "manifests" / "latest_source_manifest.json"
    lineage_path = output_root / "lineage" / "latest_lineage.json"
    run_path = output_root / "manifests" / "latest_pipeline_run.json"

    contracts = _read_json(_resolve(root, contract_path))
    _validate_contract_registry(contracts)
    current_sources = _discover_sources(source_root, root)
    previous_manifest = _read_json(manifest_path)
    changes = _compare_manifests(previous_manifest.get("sources") or [], current_sources)
    dependencies = _dependency_fingerprints(
        {
            "chunk_artifact": _resolve(root, chunk_artifact),
            "governance_artifact": _resolve(root, governance_artifact),
            "data_contract": _resolve(root, contract_path),
        },
        root,
    )
    previous_dependencies = previous_manifest.get("dependencies") or {}
    upstream_changed = sorted(
        key
        for key, value in dependencies.items()
        if previous_dependencies.get(key, {}).get("content_hash") != value["content_hash"]
    )

    for source in current_sources:
        destination = bronze_root / f"{source['content_hash']}{source['extension']}"
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_resolve(root, source["source_path"]), destination)
        source["bronze_uri"] = destination.relative_to(root).as_posix()

    source_change_detected = bool(changes["new"] or changes["changed"] or changes["deleted"])
    upstream_change_detected = bool(upstream_changed)
    materializations_exist = silver_path.exists() and gold_path.exists() and quarantine_path.exists()
    rebuild_performed = (
        source_change_detected
        or upstream_change_detected
        or not previous_manifest
        or not materializations_exist
    )

    if rebuild_performed:
        chunk_payload = _read_json(_resolve(root, chunk_artifact))
        governance_payload = _read_json(_resolve(root, governance_artifact))
        governance_index = {
            str(row.get("source_id")): row
            for row in (governance_payload.get("sources") or [])
            if row.get("source_id")
        }
        silver_rows, quarantined_rows = _build_silver_rows(
            chunk_payload.get("chunks") or [],
            governance_index,
        )
        kb_fingerprint = _fingerprint_rows(silver_rows)
        gold_rows = _build_gold_rows(silver_rows, kb_fingerprint)
        _write_jsonl(silver_path, silver_rows)
        _write_jsonl(gold_path, gold_rows)
        _write_jsonl(quarantine_path, quarantined_rows)
    else:
        silver_rows = _read_jsonl(silver_path)
        gold_rows = _read_jsonl(gold_path)
        quarantined_rows = _read_jsonl(quarantine_path)
        kb_fingerprint = _fingerprint_rows(silver_rows)

    source_manifest = {
        "schema_version": "nlcare_bronze_source_manifest_v1",
        "generated_at": _now(),
        "clinical_validation": False,
        "patient_data_allowed": False,
        "source_count": len(current_sources),
        "sources": current_sources,
        "dependencies": dependencies,
    }
    _write_json(manifest_path, source_manifest)

    quality = _quality_report(
        sources=current_sources,
        silver_rows=silver_rows,
        gold_rows=gold_rows,
        quarantined_rows=quarantined_rows,
    )
    lineage = _lineage_report(
        sources=current_sources,
        silver_rows=silver_rows,
        gold_rows=gold_rows,
        kb_fingerprint=kb_fingerprint,
        paths={
            "bronze_manifest": manifest_path.relative_to(root).as_posix(),
            "silver": silver_path.relative_to(root).as_posix(),
            "gold": gold_path.relative_to(root).as_posix(),
            "quarantine": quarantine_path.relative_to(root).as_posix(),
        },
    )
    _write_json(lineage_path, lineage)

    status = "strong" if quality["hard_failures"] == 0 else "needs_attention"
    payload = {
        "schema_version": "nlcare_data_platform_pipeline_v1",
        "generated_at": _now(),
        "status": status,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "patient_data_processed": False,
        "external_cloud_write_performed": False,
        "incremental_run": {
            **changes,
            "source_change_detected": source_change_detected,
            "upstream_changed": upstream_changed,
            "upstream_change_detected": upstream_change_detected,
            "rebuild_performed": rebuild_performed,
        },
        "layers": {
            "bronze": {
                "record_count": len(current_sources),
                "manifest": manifest_path.relative_to(root).as_posix(),
                "content_addressed": True,
            },
            "silver": {
                "record_count": len(silver_rows),
                "path": silver_path.relative_to(root).as_posix(),
                "schema": "knowledge_chunk_silver_v1",
            },
            "gold": {
                "record_count": len(gold_rows),
                "path": gold_path.relative_to(root).as_posix(),
                "schema": "vector_record_gold_v1",
                "embedding_generated": False,
                "managed_vector_upsert_performed": False,
            },
            "quarantine": {
                "record_count": len(quarantined_rows),
                "path": quarantine_path.relative_to(root).as_posix(),
            },
        },
        "quality": quality,
        "lineage": {
            "path": lineage_path.relative_to(root).as_posix(),
            "kb_fingerprint": kb_fingerprint,
            "complete": True,
        },
        "contracts": {
            "path": _resolve(root, contract_path).relative_to(root).as_posix(),
            "registry_schema": contracts["schema_version"],
            "validated": True,
        },
        "cloud_target": {
            "local_layout": "implemented",
            "azure_adls_gen2": "reference_target_not_written",
            "delta_or_parquet": "not_required_at_current_scale",
            "data_factory_or_databricks": "not_deployed",
        },
        "claim_boundary": (
            "This is an incremental pipeline over curated non-patient knowledge assets with "
            "structural identity-metadata exclusions; it is not a general PHI detector. "
            "It does not establish clinical validity, FHIR interoperability, HIPAA compliance, "
            "real-patient data quality, or production healthcare readiness."
        ),
    }
    _write_json(run_path, payload)
    return payload


def _discover_sources(source_root: Path, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not source_root.exists():
        return rows
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SOURCE_EXTENSIONS:
            continue
        stat = path.stat()
        rows.append(
            {
                "source_path": path.relative_to(root).as_posix(),
                "content_hash": _sha256_file(path),
                "size_bytes": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
                "extension": path.suffix.lower(),
            }
        )
    return rows


def _compare_manifests(
    previous: Iterable[dict[str, Any]],
    current: Iterable[dict[str, Any]],
) -> dict[str, list[str]]:
    before = {row["source_path"]: row["content_hash"] for row in previous}
    after = {row["source_path"]: row["content_hash"] for row in current}
    return {
        "new": sorted(path for path in after if path not in before),
        "changed": sorted(path for path in after if path in before and before[path] != after[path]),
        "unchanged": sorted(path for path in after if path in before and before[path] == after[path]),
        "deleted": sorted(path for path in before if path not in after),
    }


def _dependency_fingerprints(paths: dict[str, Path], root: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for key, path in paths.items():
        output[key] = {
            "path": path.relative_to(root).as_posix(),
            "content_hash": _sha256_file(path) if path.exists() else "missing",
        }
    return output


def _build_silver_rows(
    chunks: Iterable[dict[str, Any]],
    governance_index: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    silver: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in chunks:
        chunk_id = str(chunk.get("id") or "")
        parent_id = str(chunk.get("parent_id") or "")
        text = str(chunk.get("text") or "").strip()
        issues: list[str] = []
        if not chunk_id:
            issues.append("missing_chunk_id")
        if not parent_id:
            issues.append("missing_parent_id")
        if not text:
            issues.append("missing_text")
        if chunk_id in seen:
            issues.append("duplicate_chunk_id")
        seen.add(chunk_id)

        governance = governance_index.get(parent_id) or {}
        tier = str(governance.get("tier") or _tier_from_trust(chunk.get("trust_level")))
        allowed_use = list(governance.get("allowed_use") or _uses_from_trust(chunk.get("trust_level")))
        staleness = str(governance.get("staleness_status") or "unknown")
        patient_facing = tier in {"T1", "T2", "T3"} and bool(
            set(allowed_use) & {"education", "patient_safety", "monitoring_context"}
        )
        if tier == "T4" and "portal_help" in allowed_use:
            patient_facing = True

        row = {
            **chunk,
            "id": chunk_id,
            "parent_id": parent_id,
            "text": text,
            "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "source_tier": tier,
            "allowed_use": allowed_use,
            "staleness_status": staleness,
            "patient_facing": patient_facing,
            "clinical_validation": False,
            "data_contract": "knowledge_chunk_silver_v1",
        }
        if issues:
            quarantine.append({"issues": issues, "record": row})
        else:
            silver.append(row)
    return silver, quarantine


def _build_gold_rows(silver_rows: Iterable[dict[str, Any]], kb_fingerprint: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in silver_rows:
        namespace = _namespace_for(row)
        if namespace not in REMOTE_SAFE_NAMESPACES:
            continue
        metadata = {
            "source_id": row["parent_id"],
            "chunk_id": row["id"],
            "parent_id": row["parent_id"],
            "source_tier": row["source_tier"],
            "allowed_use": row["allowed_use"],
            "patient_facing": row["patient_facing"],
            "staleness_status": row["staleness_status"],
            "kb_fingerprint": kb_fingerprint,
            "doc_type": "knowledge_chunk",
            "data_scope": "curated_non_patient_kb",
            "clinical_validation": False,
            "title": row.get("title") or "Untitled source",
            "source_name": row.get("source_name") or "Local KB",
            "source_url": row.get("source_url") or "",
            "topic": row.get("topic") or "",
            "section": row.get("section") or "",
            "tags": row.get("tags") or [],
        }
        output.append(
            {
                "record_id": row["id"],
                "embedding_input": row["text"],
                "namespace": namespace,
                "metadata": metadata,
                "data_contract": "vector_record_gold_v1",
            }
        )
    return output


def _namespace_for(row: dict[str, Any]) -> str:
    if row["source_tier"] in {"T1", "T2", "T3"} and row["patient_facing"]:
        return "nlcare_kb_demo_t1_t3"
    if row["source_tier"] == "T4" and "portal_help" in row["allowed_use"]:
        return "nlcare_portal_help"
    return "local_only_quarantine"


def _quality_report(
    *,
    sources: list[dict[str, Any]],
    silver_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    quarantined_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    duplicate_sources = len(sources) - len({row["content_hash"] for row in sources})
    duplicate_chunks = len(silver_rows) - len({row["id"] for row in silver_rows})
    tier_coverage = (
        sum(row["source_tier"] != "T5" for row in silver_rows) / len(silver_rows)
        if silver_rows
        else 0.0
    )
    all_gold_metadata_complete = all(
        REQUIRED_GOLD_METADATA_KEYS.issubset(set(row.get("metadata") or {}))
        for row in gold_rows
    )
    hard_failures = (
        int(not sources)
        + int(not silver_rows)
        + int(not gold_rows)
        + int(duplicate_chunks > 0)
        + int(not all_gold_metadata_complete)
    )
    return {
        "hard_failures": hard_failures,
        "source_count": len(sources),
        "silver_record_count": len(silver_rows),
        "gold_record_count": len(gold_rows),
        "quarantined_record_count": len(quarantined_rows),
        "duplicate_source_content_count": duplicate_sources,
        "duplicate_chunk_id_count": duplicate_chunks,
        "governed_tier_coverage": round(tier_coverage, 4),
        "all_gold_metadata_excludes_banned_identity_keys": all(
            not (set(row.get("metadata") or {}) & BANNED_IDENTITY_METADATA_KEYS)
            for row in gold_rows
        ),
        "all_gold_metadata_complete": all_gold_metadata_complete,
        "all_gold_records_clinical_validation_false": all(
            row["metadata"]["clinical_validation"] is False for row in gold_rows
        ),
    }


def _lineage_report(
    *,
    sources: list[dict[str, Any]],
    silver_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    kb_fingerprint: str,
    paths: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": "nlcare_data_lineage_v1",
        "generated_at": _now(),
        "clinical_validation": False,
        "nodes": [
            {
                "id": "knowledge_sources",
                "layer": "source",
                "record_count": len(sources),
                "fingerprint": _fingerprint_rows(sources),
            },
            {
                "id": "knowledge_bronze",
                "layer": "bronze",
                "record_count": len(sources),
                "path": paths["bronze_manifest"],
            },
            {
                "id": "knowledge_silver",
                "layer": "silver",
                "record_count": len(silver_rows),
                "path": paths["silver"],
                "fingerprint": kb_fingerprint,
            },
            {
                "id": "vector_gold",
                "layer": "gold",
                "record_count": len(gold_rows),
                "path": paths["gold"],
                "fingerprint": _fingerprint_rows(gold_rows),
            },
        ],
        "edges": [
            {"from": "knowledge_sources", "to": "knowledge_bronze", "operation": "content_addressed_copy"},
            {"from": "knowledge_bronze", "to": "knowledge_silver", "operation": "chunk_validate_govern"},
            {"from": "knowledge_silver", "to": "vector_gold", "operation": "remote_safe_projection"},
        ],
        "quarantine_path": paths["quarantine"],
    }


def _validate_contract_registry(payload: dict[str, Any]) -> None:
    required_contracts = {
        "knowledge_source_bronze_v1",
        "knowledge_chunk_silver_v1",
        "vector_record_gold_v1",
    }
    if payload.get("clinical_validation") is not False:
        raise ValueError("Data contract registry must declare clinical_validation=false.")
    if payload.get("patient_data_allowed") is not False:
        raise ValueError("This pipeline cannot be configured for patient data.")
    if not required_contracts.issubset(set(payload.get("contracts") or {})):
        raise ValueError("Data contract registry is missing one or more medallion contracts.")


def _tier_from_trust(trust_level: Any) -> str:
    return {
        "clinical_safety_policy": "T1",
        "clinical_guideline_summary": "T1",
        "systematic_review": "T2",
        "research_paper": "T2",
        "patient_education": "T3",
        "local_source": "T4",
    }.get(str(trust_level or ""), "T5")


def _uses_from_trust(trust_level: Any) -> list[str]:
    return {
        "clinical_safety_policy": ["patient_safety", "education", "clinician_only"],
        "clinical_guideline_summary": ["education", "clinician_only", "monitoring_context"],
        "systematic_review": ["education", "monitoring_context"],
        "research_paper": ["education", "monitoring_context"],
        "patient_education": ["education"],
        "local_source": ["portal_help"],
    }.get(str(trust_level or ""), [])


def _fingerprint_rows(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item.get("id") or item.get("record_id") or item.get("source_path"))):
        digest.update(json.dumps(row, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "BANNED_IDENTITY_METADATA_KEYS",
    "REMOTE_SAFE_NAMESPACES",
    "REQUIRED_GOLD_METADATA_KEYS",
    "ROOT_DIR",
    "SUPPORTED_SOURCE_EXTENSIONS",
    "run_data_platform_pipeline",
]
