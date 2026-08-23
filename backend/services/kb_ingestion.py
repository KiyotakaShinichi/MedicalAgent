import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}
_INGESTED_CHUNK_CACHE = {}


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
                "source_path": _canonical_source_key(source_path),
                "source_type": source_path.suffix.lower().lstrip("."),
                "trust_level": metadata["trust_level"],
                "topic": metadata["topic"],
                "modality": metadata["modality"],
                "care_stage": metadata["care_stage"],
                "confidence": metadata["confidence"],
                "pmcid": metadata["pmcid"],
                "pmid": metadata["pmid"],
                "doi": metadata["doi"],
                "publication_date": metadata["publication_date"],
                "journal": metadata["journal"],
                "license": metadata["license"],
                "retracted": metadata["retracted"],
                "allowed_use": metadata["allowed_use"],
                "patient_facing_suitability": metadata["patient_facing_suitability"],
                "evidence_role": metadata["evidence_role"],
                "not_allowed_for": metadata["not_allowed_for"],
                "selection_rationale": metadata["selection_rationale"],
                "section": chunk["section"],
                "section_heading": chunk.get("section_heading") or chunk["section"],
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
    cache_key = str(chunk_path.resolve())
    signature = _file_signature(chunk_path)
    cached = _INGESTED_CHUNK_CACHE.get(cache_key)
    if cached and cached["signature"] == signature:
        return _copy_chunks(cached["chunks"])

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
            "source_path": chunk.get("source_path"),
            "source_type": chunk.get("source_type"),
            "tags": chunk.get("tags") or [],
            "topic": chunk.get("topic"),
            "modality": chunk.get("modality") or [],
            "care_stage": chunk.get("care_stage"),
            "section": chunk.get("section"),
            "section_heading": chunk.get("section_heading") or chunk.get("section"),
            "section_rank": chunk.get("section_rank"),
            "chunk_index": chunk.get("chunk_index"),
            "confidence": chunk.get("confidence"),
            "pmcid": chunk.get("pmcid"),
            "pmid": chunk.get("pmid"),
            "doi": chunk.get("doi"),
            "publication_date": chunk.get("publication_date"),
            "journal": chunk.get("journal"),
            "license": chunk.get("license"),
            "retracted": bool(chunk.get("retracted")),
            "allowed_use": chunk.get("allowed_use") or [],
            "patient_facing_suitability": chunk.get("patient_facing_suitability"),
            "evidence_role": chunk.get("evidence_role"),
            "not_allowed_for": chunk.get("not_allowed_for") or [],
            "selection_rationale": chunk.get("selection_rationale"),
            "ingested_at": chunk.get("ingested_at"),
            "text": chunk.get("text"),
            "trust_level": chunk.get("trust_level") or "local_source",
        })
    _INGESTED_CHUNK_CACHE[cache_key] = {
        "signature": signature,
        "chunks": normalized,
    }
    return _copy_chunks(normalized)


def clear_ingested_chunk_cache():
    _INGESTED_CHUNK_CACHE.clear()


def _file_signature(path):
    stat = path.stat()
    return {
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }


def _copy_chunks(chunks):
    output = []
    for chunk in chunks:
        cloned = dict(chunk)
        cloned["tags"] = list(chunk.get("tags") or [])
        cloned["modality"] = list(chunk.get("modality") or [])
        cloned["allowed_use"] = list(chunk.get("allowed_use") or [])
        cloned["not_allowed_for"] = list(chunk.get("not_allowed_for") or [])
        output.append(cloned)
    return output


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
    source_id_seed = pmcid or _canonical_source_key(path)
    source_id = hashlib.sha256(source_id_seed.encode("utf-8")).hexdigest()[:16]
    return {
        "source_id": source_id,
        "title": title,
        "source_name": title,
        "source_url": (
            manifest_entry.get("landing_url")
            or manifest_entry.get("pdf_url")
            or _canonical_source_key(path)
        ),
        "trust_level": trust_level,
        "topic": topic,
        "modality": modality,
        "care_stage": care_stage,
        "confidence": confidence,
        "pmcid": pmcid,
        "pmid": manifest_entry.get("pmid"),
        "doi": manifest_entry.get("doi"),
        "publication_date": manifest_entry.get("publication_date"),
        "journal": manifest_entry.get("journal"),
        "license": manifest_entry.get("license"),
        "retracted": bool(manifest_entry.get("retracted")),
        "allowed_use": list(manifest_entry.get("allowed_use") or []),
        "patient_facing_suitability": manifest_entry.get("patient_facing_suitability"),
        "evidence_role": manifest_entry.get("evidence_role"),
        "not_allowed_for": list(manifest_entry.get("not_allowed_for") or []),
        "selection_rationale": manifest_entry.get("selection_rationale"),
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
    has_named_content_section = any(
        section_name not in {"front_matter", "references"}
        for section_name, _, _ in sections
    )
    chunks = []
    for section_name, section_heading, section_text in sections:
        if section_name == "references":
            continue
        if section_name == "front_matter" and has_named_content_section:
            continue
        effective_section = "body" if section_name == "front_matter" else section_name
        for chunk_text in _chunk_text(section_text, chunk_chars, overlap_chars):
            chunks.append({
                "section": effective_section,
                "section_heading": section_heading,
                "text": f"[{effective_section}] {chunk_text}",
            })
    return chunks


def _sectionize_text(text):
    if not text:
        return []
    sections = []
    active_section = "front_matter"
    active_heading = "front_matter"
    active_lines = []
    structural_heading_seen = False

    def flush():
        body = "\n".join(active_lines).strip()
        if body:
            sections.append((active_section, active_heading, body))

    for line in text.splitlines():
        heading = _recognized_section_heading(line, structural_heading_seen=structural_heading_seen)
        if heading is None:
            active_lines.append(line)
            continue
        flush()
        active_lines = []
        active_section, active_heading = heading
        structural_heading_seen = True

    flush()
    named = [row for row in sections if row[0] not in {"front_matter", "references"}]
    if not named:
        body = "\n".join(
            body for section_name, _, body in sections
            if body and section_name != "references"
        ).strip() or text
        return [("body", "body", body)]
    return sections


def _recognized_section_heading(line, *, structural_heading_seen=False):
    raw = str(line or "").strip().strip("#").strip()
    if not raw or len(raw) > 120:
        return None
    normalized = re.sub(r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?\s+", "", raw, flags=re.IGNORECASE)
    normalized = normalized.rstrip(":").strip()
    lower = re.sub(r"\s+", " ", normalized.lower())
    direct = {
        "abstract": "abstract",
        "highlights": "abstract",
        "summary": "abstract",
        "plain language summary": "abstract",
        "key points": "abstract",
        "introduction": "introduction",
        "background": "introduction",
        "methods": "methods",
        "methodology": "methods",
        "materials and methods": "methods",
        "patients and methods": "methods",
        "study design": "methods",
        "statistical analysis": "methods",
        "results": "results",
        "findings": "results",
        "outcomes": "results",
        "discussion": "discussion",
        "limitations": "discussion",
        "strengths and limitations": "discussion",
        "conclusion": "conclusion",
        "conclusions": "conclusion",
        "recommendations": "conclusion",
        "clinical implications": "conclusion",
        "references": "references",
        "bibliography": "references",
    }
    if lower in direct:
        return direct[lower], normalized
    if not structural_heading_seen:
        return None
    thematic_prefixes = (
        "incidence and prevalence",
        "prevalence",
        "anxiety",
        "depression",
        "risk factors",
        "classification and diagnosis",
        "screening",
        "assessment",
        "management",
        "treatment",
        "implementation",
        "follow-up",
        "future directions",
    )
    if any(lower == prefix or lower.startswith(prefix + " ") for prefix in thematic_prefixes):
        return "body", normalized
    return None


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


def _extract_pmcid(path, _text):
    """Return a source identity only when the source file owns that identity.

    Curated summaries often link to several PMC papers.  Treating the first
    embedded link as the summary's PMCID collapses two distinct sources and
    makes citation provenance look stronger than it is.  Research downloads
    receive their PMCID from the source manifest or their filename; linked
    identifiers inside arbitrary document text are references, not identity.
    """
    match = re.search(r"PMC\d{5,}", path.name, flags=re.IGNORECASE)
    if match:
        return match.group(0).upper()
    return None


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


def _canonical_source_key(path):
    """Platform-independent string used to derive KB source and chunk ids.

    `str(Path(...))` renders the separator of whatever OS is running, so the
    same source file produced different ids on Windows and Linux:

        Windows  KnowledgeBase\\raw\\...\\minimum_evidence...md -> 28cfcee61ce1e4a4
        Linux    KnowledgeBase/raw/.../minimum_evidence...md    -> 191dafae170c06c0

    Those ids are the key the KB source-governance map is looked up by, so on
    Linux every ingested chunk resolved to no governance entry, arrived at the
    pre-generation tier filter with no tier or allowed_use, and was correctly
    dropped - taking retrieval_context, citations, and the regression pass rate
    with it. Only the hardcoded seed snippets, whose ids are static, survived.

    Backslashes are canonical here, not because they are a good choice, but
    because they are the form the established identifiers already encode:
    twenty-one committed evidence artifacts key on these ids, including a
    frozen claim-selector holdout bank. Re-issuing every identifier to adopt
    POSIX separators would invalidate all of them, which is a far larger and
    less reversible change than pinning the convention already in use. The
    point is that the id no longer depends on which OS ran the ingestion.
    """
    return str(path).replace("/", "\\")


def _chunk_id(path, index, chunk_text):
    digest = hashlib.sha256(
        f"{_canonical_source_key(path)}:{index}:{chunk_text[:80]}".encode("utf-8")
    ).hexdigest()
    return digest[:20]


def _clean_text(text):
    text = re.sub(r"\r\n?", "\n", text or "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
