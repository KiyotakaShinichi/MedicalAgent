import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def ingest_knowledge_base(
    input_dir="KnowledgeBase/raw",
    output_path="Data/rag_knowledge_base_chunks.json",
    chunk_chars=1400,
    overlap_chars=180,
):
    source_dir = Path(input_dir)
    output = Path(output_path)
    source_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    source_files = [
        path for path in sorted(source_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    chunks = []
    skipped = []
    for source_path in source_files:
        try:
            text = _extract_text(source_path)
        except ValueError as exc:
            skipped.append({"path": str(source_path), "reason": str(exc)})
            continue
        metadata = _source_metadata(source_path, text)
        source_chunks = _chunk_text(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        for index, chunk_text in enumerate(source_chunks):
            chunk_id = _chunk_id(source_path, index, chunk_text)
            chunks.append({
                "id": chunk_id,
                "parent_id": metadata["source_id"],
                "title": metadata["title"],
                "source_name": metadata["source_name"],
                "source_url": metadata["source_url"],
                "source_path": str(source_path),
                "source_type": source_path.suffix.lower().lstrip("."),
                "trust_level": metadata["trust_level"],
                "tags": metadata["tags"],
                "chunk_index": index,
                "text": chunk_text,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            })

    payload = {
        "schema_version": "rag_knowledge_base_chunks_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_count": len(source_files),
        "chunk_count": len(chunks),
        "skipped": skipped,
        "chunks": chunks,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "input_dir": str(source_dir),
        "output_path": str(output),
        "source_count": len(source_files),
        "chunk_count": len(chunks),
        "skipped_count": len(skipped),
        "skipped": skipped,
    }


def load_ingested_chunks(path="Data/rag_knowledge_base_chunks.json"):
    chunk_path = Path(path)
    if not chunk_path.exists():
        return []
    payload = json.loads(chunk_path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks") if isinstance(payload, dict) else payload
    if not isinstance(chunks, list):
        return []
    normalized = []
    for chunk in chunks:
        if not isinstance(chunk, dict) or not chunk.get("text"):
            continue
        normalized.append({
            "id": str(chunk.get("id")),
            "parent_id": str(chunk.get("parent_id") or chunk.get("id")),
            "title": chunk.get("title") or "Untitled source",
            "source_name": chunk.get("source_name") or "Local KB",
            "source_url": chunk.get("source_url") or chunk.get("source_path") or "KnowledgeBase/raw",
            "tags": chunk.get("tags") or [],
            "text": chunk.get("text"),
            "trust_level": chunk.get("trust_level") or "local_source",
        })
    return normalized


def _extract_text(path):
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return _clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError("pypdf is required for PDF ingestion") from exc
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return _clean_text("\n\n".join(pages))
    raise ValueError(f"Unsupported file type: {suffix}")


def _source_metadata(path, text):
    title = _title_from_text(path, text)
    tags = _infer_tags(f"{path.name} {title} {text[:2000]}")
    trust_level = _infer_trust_level(path)
    source_id = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    return {
        "source_id": source_id,
        "title": title,
        "source_name": title,
        "source_url": str(path),
        "trust_level": trust_level,
        "tags": tags,
    }


def _chunk_text(text, chunk_chars, overlap_chars):
    if not text:
        return []
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    chunks = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= chunk_chars:
            current = paragraph
        else:
            chunks.extend(_split_long_text(paragraph, chunk_chars, overlap_chars))
            current = ""
    if current:
        chunks.append(current)
    return chunks


def _split_long_text(text, chunk_chars, overlap_chars):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return [chunk for chunk in chunks if chunk]


def _title_from_text(path, text):
    for line in text.splitlines():
        cleaned = line.strip(" #\t")
        if cleaned:
            return cleaned[:160]
    return path.stem.replace("_", " ").replace("-", " ").title()


def _infer_tags(text):
    lower = text.lower()
    tag_rules = {
        "breast cancer": ["breast", "brca"],
        "chemotherapy": ["chemotherapy", "chemo", "paclitaxel", "doxorubicin", "cyclophosphamide"],
        "mri": ["mri", "dce", "imaging", "radiology"],
        "cbc": ["cbc", "wbc", "hemoglobin", "platelets", "neutrophil"],
        "toxicity": ["toxicity", "adverse", "side effect", "neutropenia"],
        "response": ["response", "pcr", "pathologic complete response", "tumor"],
        "guideline": ["guideline", "recommendation", "consensus"],
        "patient education": ["patient", "symptom", "care team"],
    }
    tags = [tag for tag, terms in tag_rules.items() if any(term in lower for term in terms)]
    return tags or ["local_kb"]


def _infer_trust_level(path):
    lower = str(path).lower()
    if "guideline" in lower or "nccn" in lower or "asco" in lower:
        return "clinical_guideline"
    if "paper" in lower or "pubmed" in lower or "journal" in lower:
        return "research_paper"
    if "patient" in lower or "education" in lower:
        return "patient_education"
    return "local_source"


def _chunk_id(path, index, chunk_text):
    digest = hashlib.sha256(f"{path}:{index}:{chunk_text[:80]}".encode("utf-8")).hexdigest()
    return digest[:20]


def _clean_text(text):
    text = re.sub(r"\r\n?", "\n", text or "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
