import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def ingest_knowledge_base(
    input_dir="KnowledgeBase/raw",
    output_path="Data/rag_knowledge_base_chunks.json",
    chunk_chars=2200,
    overlap_chars=220,
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
    source_manifest = _load_source_manifest(source_dir)
    for source_path in source_files:
        try:
            text = _extract_text(source_path)
        except ValueError as exc:
            skipped.append({"path": str(source_path), "reason": str(exc)})
            continue
        metadata = _source_metadata(source_path, text, source_manifest)
        source_chunks = _chunk_text_by_section(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        for index, chunk in enumerate(source_chunks):
            chunk_text = chunk["text"]
            chunk_id = _chunk_id(source_path, index, chunk_text)
            tags = sorted(set(metadata["tags"] + _infer_tags(f"{chunk['section']} {chunk_text[:1600]}")))
            chunks.append({
                "id": chunk_id,
                "parent_id": metadata["source_id"],
                "title": metadata["title"],
                "source_name": metadata["source_name"],
                "source_url": metadata["source_url"],
                "source_path": str(source_path),
                "source_type": source_path.suffix.lower().lstrip("."),
                "trust_level": metadata["trust_level"],
                "topic": metadata["topic"],
                "modality": metadata["modality"],
                "care_stage": metadata["care_stage"],
                "confidence": metadata["confidence"],
                "pmcid": metadata["pmcid"],
                "section": chunk["section"],
                "section_rank": _section_rank(chunk["section"]),
                "tags": tags,
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
        "quality_checks": _kb_quality_checks(chunks),
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
            "topic": chunk.get("topic"),
            "modality": chunk.get("modality") or [],
            "care_stage": chunk.get("care_stage"),
            "section": chunk.get("section"),
            "confidence": chunk.get("confidence"),
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


def _source_metadata(path, text, source_manifest=None):
    manifest_entry = _manifest_entry_for_path(path, source_manifest or {})
    title = _title_from_text(path, text)
    title = manifest_entry.get("title") or title
    topic = manifest_entry.get("topic") or _infer_topic(f"{path.name} {title} {text[:2000]}")
    modality = manifest_entry.get("modality") or _infer_modality(f"{path.name} {title} {text[:2000]}")
    care_stage = manifest_entry.get("stage") or _infer_care_stage(f"{path.name} {title} {text[:2000]}")
    confidence = manifest_entry.get("confidence") or _infer_confidence(path)
    trust_level = manifest_entry.get("trust_level") or _infer_trust_level(path)
    pmcid = manifest_entry.get("pmcid") or _extract_pmcid(path, text)
    tags = sorted(set(_infer_tags(f"{path.name} {title} {topic} {' '.join(modality)} {care_stage} {text[:2000]}")))
    source_id_seed = pmcid or str(path)
    source_id = hashlib.sha256(source_id_seed.encode("utf-8")).hexdigest()[:16]
    return {
        "source_id": source_id,
        "title": title,
        "source_name": title,
        "source_url": manifest_entry.get("landing_url") or manifest_entry.get("pdf_url") or str(path),
        "trust_level": trust_level,
        "topic": topic,
        "modality": modality,
        "care_stage": care_stage,
        "confidence": confidence,
        "pmcid": pmcid,
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


def _chunk_text_by_section(text, chunk_chars, overlap_chars):
    sections = _sectionize_text(text)
    chunks = []
    for section_name, section_text in sections:
        if section_name in {"front_matter", "references"}:
            continue
        for chunk_text in _chunk_text(section_text, chunk_chars, overlap_chars):
            chunks.append({
                "section": section_name,
                "text": f"[{section_name}] {chunk_text}",
            })
    return chunks


def _sectionize_text(text):
    if not text:
        return []
    section_names = [
        "abstract",
        "introduction",
        "background",
        "methods",
        "materials and methods",
        "patients and methods",
        "results",
        "discussion",
        "conclusion",
        "conclusions",
        "clinical implications",
        "references",
    ]
    pattern = re.compile(
        r"(?im)^\s*(abstract|introduction|background|methods|materials and methods|patients and methods|results|discussion|conclusions?|clinical implications|references)\s*$"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return [("body", text)]
    sections = []
    if matches[0].start() > 0:
        sections.append(("front_matter", text[:matches[0].start()].strip()))
    for index, match in enumerate(matches):
        section = match.group(1).lower()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append((section, section_text))
    return [(name, body) for name, body in sections if body]


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
        "radiomics": ["radiomics", "texture", "heterogeneity", "feature"],
        "machine learning": ["machine learning", "classifier", "prediction", "model"],
        "clinical safety": ["fever", "febrile", "emergency", "urgent"],
    }
    tags = [tag for tag, terms in tag_rules.items() if any(term in lower for term in terms)]
    return tags or ["local_kb"]


def _infer_topic(text):
    lower = text.lower()
    if any(term in lower for term in ["neutropenia", "febrile", "wbc", "anc", "hematologic"]):
        return "cbc_toxicity_monitoring"
    if any(term in lower for term in ["mri", "dce", "radiomics", "texture", "heterogeneity"]):
        return "mri_response_monitoring"
    if any(term in lower for term in ["pcr", "pathologic complete response", "neoadjuvant", "response prediction"]):
        return "chemotherapy_response_prediction"
    if any(term in lower for term in ["fever", "symptom", "adverse event"]):
        return "treatment_safety"
    return "general_breast_cancer_monitoring"


def _infer_modality(text):
    lower = text.lower()
    modalities = []
    if any(term in lower for term in ["mri", "dce", "diffusion", "radiomics"]):
        modalities.append("MRI")
    if any(term in lower for term in ["cbc", "wbc", "anc", "hemoglobin", "platelet", "neutropenia"]):
        modalities.append("CBC")
    if any(term in lower for term in ["symptom", "fever", "fatigue", "nausea", "pain"]):
        modalities.append("symptoms")
    if any(term in lower for term in ["chemotherapy", "neoadjuvant", "treatment", "regimen"]):
        modalities.append("treatment")
    return modalities or ["clinical"]


def _infer_care_stage(text):
    lower = text.lower()
    if "neoadjuvant" in lower:
        return "neoadjuvant_treatment"
    if any(term in lower for term in ["toxicity", "neutropenia", "fever", "adverse"]):
        return "treatment_toxicity_monitoring"
    if any(term in lower for term in ["follow-up", "follow up", "survival", "recurrence"]):
        return "follow_up"
    return "treatment_monitoring"


def _infer_confidence(path):
    lower = str(path).lower()
    if "pmc" in lower or "research_papers" in lower:
        return "peer_reviewed_open_access"
    if "guideline" in lower:
        return "clinical_guideline"
    return "local_source"


def _infer_trust_level(path):
    lower = str(path).lower()
    if "guideline" in lower or "nccn" in lower or "asco" in lower:
        return "clinical_guideline"
    if "paper" in lower or "pubmed" in lower or "journal" in lower:
        return "research_paper"
    if "patient" in lower or "education" in lower:
        return "patient_education"
    return "local_source"


def _section_rank(section):
    ranks = {
        "abstract": 1,
        "conclusion": 2,
        "conclusions": 2,
        "clinical implications": 3,
        "results": 4,
        "methods": 5,
        "materials and methods": 5,
        "patients and methods": 5,
        "discussion": 6,
        "introduction": 7,
        "background": 8,
        "body": 9,
        "front_matter": 10,
        "references": 99,
    }
    return ranks.get(section, 50)


def _extract_pmcid(path, text):
    match = re.search(r"PMC\d{5,}", path.name, flags=re.IGNORECASE)
    if match:
        return match.group(0).upper()
    match = re.search(r"\bPMC\d{5,}\b", text[:3000], flags=re.IGNORECASE)
    return match.group(0).upper() if match else None


def _load_source_manifest(source_dir):
    manifests = list(source_dir.rglob("research_papers_manifest.json"))
    entries = {}
    for manifest in manifests:
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for item in payload.get("items") or []:
            if item.get("file_name"):
                entries[item["file_name"]] = item
            if item.get("path"):
                entries[str(Path(item["path"]).name)] = item
    return entries


def _manifest_entry_for_path(path, source_manifest):
    return source_manifest.get(path.name) or {}


def _kb_quality_checks(chunks):
    strong_claim_terms = ["always", "never", "proves", "guarantees", "cures", "replaces pathology"]
    opposing_terms = [
        ("accurate", "not accurate"),
        ("high sensitivity", "low sensitivity"),
        ("overestimate", "underestimate"),
        ("predicts pcr", "cannot predict pcr"),
    ]
    strong_claims = []
    by_topic = {}
    for chunk in chunks:
        text = chunk.get("text", "").lower()
        if any(term in text for term in strong_claim_terms):
            strong_claims.append({
                "id": chunk.get("id"),
                "title": chunk.get("title"),
                "section": chunk.get("section"),
            })
        by_topic.setdefault(chunk.get("topic") or "unknown", []).append(chunk)
    contradiction_watchlist = []
    for topic, topic_chunks in by_topic.items():
        joined = " ".join(chunk.get("text", "").lower() for chunk in topic_chunks)
        for positive, negative in opposing_terms:
            if positive in joined and negative in joined:
                contradiction_watchlist.append({
                    "topic": topic,
                    "signal": f"{positive} vs {negative}",
                    "note": "Heuristic watchlist only; use an NLI/LLM evaluator later.",
                })
    return {
        "strong_claim_watchlist": strong_claims[:20],
        "contradiction_watchlist": contradiction_watchlist[:20],
        "section_counts": _counts(chunk.get("section") or "unknown" for chunk in chunks),
        "topic_counts": _counts(chunk.get("topic") or "unknown" for chunk in chunks),
    }


def _counts(values):
    output = {}
    for value in values:
        output[value] = output.get(value, 0) + 1
    return output


def _chunk_id(path, index, chunk_text):
    digest = hashlib.sha256(f"{path}:{index}:{chunk_text[:80]}".encode("utf-8")).hexdigest()
    return digest[:20]


def _clean_text(text):
    text = re.sub(r"\r\n?", "\n", text or "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
