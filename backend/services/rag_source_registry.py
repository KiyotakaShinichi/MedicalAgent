import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_KB_CHUNKS_PATH = "Data/rag_knowledge_base_chunks.json"


def build_rag_source_registry(path=DEFAULT_KB_CHUNKS_PATH, source_limit=25):
    payload = _load_payload(path)
    chunks = payload.get("chunks") or []
    sources = defaultdict(list)
    for chunk in chunks:
        source_key = chunk.get("parent_id") or chunk.get("source_name") or chunk.get("source_path") or chunk.get("id")
        sources[source_key].append(chunk)

    source_rows = []
    for source_id, source_chunks in sources.items():
        first = source_chunks[0]
        source_rows.append({
            "source_id": source_id,
            "title": first.get("title") or first.get("source_name") or "Untitled source",
            "source_url": first.get("source_url"),
            "source_path": first.get("source_path"),
            "pmcid": first.get("pmcid"),
            "trust_level": first.get("trust_level") or "unknown",
            "confidence": first.get("confidence") or "unknown",
            "chunk_count": len(source_chunks),
            "topics": sorted({chunk.get("topic") or "unknown" for chunk in source_chunks}),
            "modalities": sorted({
                modality
                for chunk in source_chunks
                for modality in (chunk.get("modality") or ["unknown"])
            }),
            "sections": _counts(chunk.get("section") or "unknown" for chunk in source_chunks),
        })

    source_rows = sorted(source_rows, key=lambda row: row["chunk_count"], reverse=True)
    quality_checks = payload.get("quality_checks") or {}
    return {
        "status": _registry_status(payload, source_rows, quality_checks),
        "purpose": "RAG source registry for KB governance, source trust, citation traceability, and ingestion quality review.",
        "path": str(path),
        "schema_version": payload.get("schema_version"),
        "generated_at": payload.get("generated_at"),
        "source_count": len(source_rows),
        "chunk_count": len(chunks),
        "topic_counts": quality_checks.get("topic_counts") or _counts(chunk.get("topic") or "unknown" for chunk in chunks),
        "modality_counts": _modality_counts(chunks),
        "confidence_counts": _counts(chunk.get("confidence") or "unknown" for chunk in chunks),
        "trust_level_counts": _counts(chunk.get("trust_level") or "unknown" for chunk in chunks),
        "section_counts": quality_checks.get("section_counts") or _counts(chunk.get("section") or "unknown" for chunk in chunks),
        "quality_checks": {
            "strong_claim_watchlist": quality_checks.get("strong_claim_watchlist") or [],
            "contradiction_watchlist": quality_checks.get("contradiction_watchlist") or [],
        },
        "citation_policy": [
            "Every RAG answer using retrieved context should expose citations.",
            "Patient-specific answers should not be cached or generalized across patients.",
            "Strong clinical claims require source support and clinician-review language.",
            "Research papers improve educational retrieval but do not create clinical validation.",
        ],
        "sources": source_rows[:source_limit],
        "limitations": [
            "This registry is derived from local ingestion metadata, not a formal clinical guideline library.",
            "Contradiction and strong-claim checks are lightweight heuristics until a labeled evaluator is added.",
        ],
    }


def _registry_status(payload, sources, quality_checks):
    if not payload or not sources:
        return "unavailable"
    if quality_checks.get("contradiction_watchlist"):
        return "acceptable"
    if len(sources) < 3:
        return "unideal"
    return "passed"


def _load_payload(path):
    chunk_path = Path(path)
    if not chunk_path.exists():
        return {}
    try:
        return json.loads(chunk_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _modality_counts(chunks):
    counter = Counter()
    for chunk in chunks:
        for modality in chunk.get("modality") or ["unknown"]:
            counter[str(modality)] += 1
    return dict(counter)


def _counts(values):
    return dict(Counter(str(value) for value in values))
