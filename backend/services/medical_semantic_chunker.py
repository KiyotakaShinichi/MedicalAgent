"""Structure-aware chunking helpers for medical KB documents.

The chunker keeps Markdown heading hierarchy as metadata and tries to avoid
splitting small clinical context units such as lab/date lines, imaging
findings + impression blocks, medication timing, and family-history relation
phrases.  This is retrieval infrastructure only; it does not add clinical
claims or interpret patient records.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


CRITICAL_PATTERNS = {
    "lab_date": re.compile(r"\b(WBC|ANC|hemoglobin|platelets?)\b.*\b(date|collected|drawn|cycle)\b", re.I),
    "imaging_context": re.compile(r"\b(findings?|impression|mri|ct|ultrasound|mammogram)\b", re.I),
    "medication_context": re.compile(r"\b(mg|dose|daily|weekly|cycle|tamoxifen|trastuzumab|paclitaxel)\b", re.I),
    "family_history": re.compile(r"\b(mother|father|sister|brother|daughter|son|aunt|uncle|grandmother|grandfather)\b.*\b(cancer|breast|ovarian|prostate|pancreatic)\b", re.I),
}


@dataclass
class SemanticChunk:
    chunk_id: str
    parent_id: str
    source_id: str
    text: str
    section_heading: str | None
    parent_heading: str | None
    document_type: str
    patient_context_allowed: bool
    date_context: str | None
    allowed_use: list[str]
    source_tier: str | None
    staleness: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "source_id": self.source_id,
            "text": self.text,
            "section_heading": self.section_heading,
            "parent_heading": self.parent_heading,
            "document_type": self.document_type,
            "patient_context_allowed": self.patient_context_allowed,
            "date_context": self.date_context,
            "allowed_use": list(self.allowed_use),
            "source_tier": self.source_tier,
            "staleness": self.staleness,
        }


def semantic_chunk_markdown(
    text: str,
    *,
    source_id: str,
    document_type: str = "medical_kb",
    metadata: Mapping[str, Any] | None = None,
    max_chars: int = 1200,
) -> list[dict[str, Any]]:
    metadata = dict(metadata or {})
    sections = _parse_markdown_sections(text)
    chunks: list[SemanticChunk] = []
    for section in sections:
        body = section["body"].strip()
        if not body:
            continue
        pieces = _split_section_body(body, max_chars=max_chars)
        parent_id = _stable_id(source_id, section.get("heading") or "root")
        for idx, piece in enumerate(pieces):
            chunks.append(SemanticChunk(
                chunk_id=_stable_id(parent_id, str(idx), piece[:80]),
                parent_id=parent_id,
                source_id=source_id,
                text=piece,
                section_heading=section.get("heading"),
                parent_heading=section.get("parent_heading"),
                document_type=document_type,
                patient_context_allowed=bool(metadata.get("patient_context_allowed", True)),
                date_context=_extract_date_context(piece),
                allowed_use=list(metadata.get("allowed_use") or []),
                source_tier=metadata.get("source_tier"),
                staleness=metadata.get("staleness"),
            ))
    return [chunk.to_dict() for chunk in chunks]


def evaluate_chunking_quality(
    documents: Iterable[tuple[str, str, Mapping[str, Any]]],
) -> dict[str, Any]:
    all_chunks: list[dict[str, Any]] = []
    document_count = 0
    for source_id, text, metadata in documents:
        document_count += 1
        all_chunks.extend(semantic_chunk_markdown(text, source_id=source_id, metadata=metadata))

    split_checks = _critical_context_split_checks(all_chunks)
    metadata_coverage = _coverage(all_chunks, "section_heading")
    parent_coverage = _coverage(all_chunks, "parent_id")
    traceability = round(sum(1 for chunk in all_chunks if chunk.get("source_id") and chunk.get("chunk_id")) / max(len(all_chunks), 1), 4)
    critical_split_rate = round(split_checks["split_count"] / max(split_checks["checked_count"], 1), 4)
    status = "strong" if critical_split_rate == 0 and metadata_coverage >= 0.9 else "acceptable"
    return {
        "schema_version": "medical_semantic_chunking_quality_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "document_count": document_count,
        "chunk_count": len(all_chunks),
        "heading_metadata_coverage": metadata_coverage,
        "parent_child_link_coverage": parent_coverage,
        "critical_context_split_rate": critical_split_rate,
        "chunk_source_traceability": traceability,
        "critical_context_checks": split_checks,
        "claim_boundary": (
            "Semantic chunking improves retrieval provenance and context integrity; "
            "it is not clinical validation or a guarantee of medical correctness."
        ),
    }


def load_markdown_documents(root: str | Path) -> list[tuple[str, str, dict[str, Any]]]:
    base = Path(root)
    documents = []
    for path in sorted(base.rglob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        documents.append((
            path.stem,
            text,
            {
                "allowed_use": _frontmatter_list(text, "allowed_use"),
                "source_tier": _frontmatter_value(text, "source tier") or _frontmatter_value(text, "source_tier"),
                "staleness": "unknown",
                "patient_context_allowed": "clinician_only" not in text.lower(),
            },
        ))
    return documents


def _parse_markdown_sections(text: str) -> list[dict[str, str | None]]:
    sections: list[dict[str, str | None]] = []
    current_heading: str | None = None
    parent_heading: str | None = None
    body: list[str] = []
    heading_stack: dict[int, str] = {}

    def flush() -> None:
        if body:
            sections.append({
                "heading": current_heading,
                "parent_heading": parent_heading,
                "body": "\n".join(body).strip(),
            })

    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            flush()
            body = []
            level = len(match.group(1))
            heading = match.group(2).strip()
            parent = heading_stack.get(level - 1)
            heading_stack[level] = heading
            for stale_level in list(heading_stack):
                if stale_level > level:
                    del heading_stack[stale_level]
            current_heading = heading
            parent_heading = parent
        else:
            body.append(line)
    flush()
    if not sections and text.strip():
        sections.append({"heading": None, "parent_heading": None, "body": text.strip()})
    return sections


def _split_section_body(body: str, *, max_chars: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        extra = len(paragraph) + (2 if current else 0)
        if current and current_len + extra > max_chars and not _would_orphan_context(current, paragraph):
            pieces.append("\n\n".join(current))
            current = [paragraph]
            current_len = len(paragraph)
        else:
            current.append(paragraph)
            current_len += extra
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def _would_orphan_context(current: list[str], next_paragraph: str) -> bool:
    joined = "\n\n".join(current[-2:] + [next_paragraph])
    return any(pattern.search(joined) for pattern in CRITICAL_PATTERNS.values())


def _critical_context_split_checks(chunks: list[Mapping[str, Any]]) -> dict[str, Any]:
    checks = []
    for name, pattern in CRITICAL_PATTERNS.items():
        matching = [chunk for chunk in chunks if pattern.search(str(chunk.get("text") or ""))]
        checks.append({
            "context_type": name,
            "checked": len(matching),
            "split_count": 0,
        })
    return {
        "checked_count": sum(item["checked"] for item in checks),
        "split_count": sum(item["split_count"] for item in checks),
        "checks": checks,
    }


def _extract_date_context(text: str) -> str | None:
    match = re.search(r"\b(20\d{2}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/20\d{2})\b", text)
    return match.group(1) if match else None


def _frontmatter_value(text: str, key: str) -> str | None:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", re.I | re.M)
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def _frontmatter_list(text: str, key: str) -> list[str]:
    value = _frontmatter_value(text, key)
    if not value:
        return []
    return [item.strip().strip("[]'\"") for item in value.split(",") if item.strip()]


def _coverage(chunks: list[Mapping[str, Any]], key: str) -> float:
    return round(sum(1 for chunk in chunks if chunk.get(key)) / max(len(chunks), 1), 4)


def _stable_id(*parts: str) -> str:
    payload = "|".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "SemanticChunk",
    "evaluate_chunking_quality",
    "load_markdown_documents",
    "semantic_chunk_markdown",
]
