"""Evidence and claim reference projection without raw passage retention."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from backend.services.rag_evidence.utilities import chunk_id, safe_float


def evidence_references(
    chunks: Sequence[Mapping[str, Any]],
    decision_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    items: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for chunk in chunks:
        current_chunk_id = chunk_id(chunk)
        decision = decision_by_id.get(current_chunk_id, {})
        source_id = str(decision.get("source_id") or chunk.get("source_id") or chunk.get("parent_id") or "")
        tier = str(decision.get("tier") or chunk.get("source_tier") or chunk.get("tier") or "")
        allowed_use = decision.get("allowed_use") or chunk.get("allowed_use") or []
        if isinstance(allowed_use, str):
            allowed_use = [allowed_use]
        staleness = str(
            decision.get("staleness_status")
            or chunk.get("staleness_status")
            or chunk.get("staleness")
            or "unknown"
        )
        item = {
            "chunk_id": current_chunk_id,
            "source_id": source_id,
            "tier": tier,
            "allowed_use": sorted(str(value) for value in allowed_use),
            "staleness_status": staleness,
            "reference": str(chunk.get("source_url") or chunk.get("source_path") or current_chunk_id),
        }
        items.append(item)
        metadata.append({
            **item,
            "title": str(chunk.get("title") or chunk.get("source_name") or ""),
            "source_name": str(chunk.get("source_name") or ""),
        })
        references.append({
            "chunk_id": current_chunk_id,
            "source_id": source_id,
            "title": str(chunk.get("title") or ""),
            "reference": item["reference"],
        })
    return items, metadata, references


def claim_references(
    claim_validation: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    claims: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    for index, verdict in enumerate(claim_validation.get("verdicts") or []):
        if not isinstance(verdict, Mapping) or verdict.get("is_claim") is False:
            continue
        sentence = str(verdict.get("sentence") or "")
        claim_id = f"claim-{index + 1}-{hashlib.sha256(sentence.encode('utf-8')).hexdigest()[:12]}"
        source_ids = [str(value) for value in verdict.get("supporting_chunk_ids") or [] if value]
        claims.append({
            "claim_id": claim_id,
            "claim_hash": hashlib.sha256(sentence.encode("utf-8")).hexdigest(),
            "claim_type": str(verdict.get("claim_type") or "unknown"),
            "status": str(verdict.get("status") or "unknown"),
            "support_score": safe_float(verdict.get("support_score")),
            "validation_method": str(verdict.get("validation_method") or "unknown"),
        })
        mappings.append({
            "claim_id": claim_id,
            "source_chunk_ids": source_ids,
            "status": str(verdict.get("status") or "unknown"),
        })
    return claims, mappings
