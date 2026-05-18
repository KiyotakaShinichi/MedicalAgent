"""Claim-level citation validation for RAG replies.

Default mode is a fast token-overlap validator. Set
``ONCOTRACK_RAG_CLAIM_VALIDATOR=nli`` to add an optional entailment/NLI pass
over the best retrieved chunks. If the NLI dependency or model is unavailable,
the service falls back to the heuristic and records that fallback in each
verdict. This keeps CI lightweight while making the production path upgradeable.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Mapping


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

_CLAIM_SIGNALS = (
    "wbc", "anc", "hemoglobin", "platelets", "neutropenia", "neutrophil",
    "tumor", "metastat", "lesion", "biopsy", "biomarker", "her2", "er+",
    "pr+", "brca", "ki-67", "ki67", "tumor marker", "ca 15-3", "ca 27", "cea",
    "chemo", "chemotherapy", "doxorubicin", "paclitaxel", "trastuzumab",
    "tamoxifen", "cyclophosphamide", "carboplatin", "docetaxel",
    "mg", "milligram", "mri", "ct ", "ultrasound", "mammogram",
    "stage i", "stage ii", "stage iii", "stage iv",
    "diagnosis", "prognosis", "survival", "recurrence", "remission",
    "side effect", "side-effect", "interaction", "contraindicated",
    "recommend", "recommended", "should take", "dose", "dosage", "increase", "decrease",
    "st. john", "john's wort", "supplement",
    "percent", "%", " 5 ", " 10 ", " 20 ", " 50 ",
)

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "then", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "for", "of", "in", "on", "to", "from", "by", "with",
    "as", "at", "this", "that", "these", "those", "it", "its", "their",
    "your", "you", "we", "our", "i", "me", "my", "they", "them", "he",
    "she", "his", "her", "not", "no", "yes", "can", "could", "should",
    "would", "may", "might", "will", "than", "such", "about", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "again", "further", "more", "most", "other", "some", "any", "all",
    "each", "few", "many", "much",
})

SUPPORTED_THRESHOLD = 0.30
WEAKLY_SUPPORTED_THRESHOLD = 0.12
NLI_ENTAILMENT_THRESHOLD = 0.58
NLI_CONTRADICTION_THRESHOLD = 0.62
VALIDATOR_ENV = "ONCOTRACK_RAG_CLAIM_VALIDATOR"
NLI_MODEL_ENV = "ONCOTRACK_RAG_NLI_MODEL"


@dataclass
class ClaimVerdict:
    sentence: str
    is_claim: bool
    support_score: float
    status: str
    supporting_chunk_ids: list[str] = field(default_factory=list)
    reason: str | None = None
    validation_method: str = "heuristic_overlap"
    entailment_score: float | None = None
    contradiction_score: float | None = None

    def to_dict(self) -> dict:
        return {
            "sentence": self.sentence,
            "is_claim": self.is_claim,
            "support_score": round(self.support_score, 4),
            "status": self.status,
            "supporting_chunk_ids": list(self.supporting_chunk_ids),
            "reason": self.reason,
            "validation_method": self.validation_method,
            "entailment_score": round(self.entailment_score, 4) if self.entailment_score is not None else None,
            "contradiction_score": round(self.contradiction_score, 4) if self.contradiction_score is not None else None,
        }


@dataclass
class ClaimValidationResult:
    verdicts: list[ClaimVerdict] = field(default_factory=list)
    claim_count: int = 0
    supported_count: int = 0
    weakly_supported_count: int = 0
    unsupported_count: int = 0
    claim_support_rate: float = 0.0
    citation_status: str = "complete"
    validation_method: str = "heuristic_overlap"
    nli_available: bool = False

    def to_dict(self) -> dict:
        return {
            "claim_count": self.claim_count,
            "supported_count": self.supported_count,
            "weakly_supported_count": self.weakly_supported_count,
            "unsupported_count": self.unsupported_count,
            "claim_support_rate": round(self.claim_support_rate, 4),
            "citation_status": self.citation_status,
            "validation_method": self.validation_method,
            "nli_available": self.nli_available,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


def validate_claims(
    reply: str,
    retrieved_chunks: Iterable[Mapping[str, object]],
    *,
    method: str | None = None,
) -> ClaimValidationResult:
    result = ClaimValidationResult()
    selected_method = _selected_method(method)
    result.validation_method = selected_method
    if not reply or not reply.strip():
        result.citation_status = "missing"
        return result

    chunks = [c for c in retrieved_chunks if c and isinstance(c, Mapping)]
    chunk_records = [
        {"id": _chunk_id(c), "text": str(c.get("text") or ""), "tokens": _tokens(str(c.get("text") or ""))}
        for c in chunks
    ]
    for sentence in _split_sentences(reply):
        result.verdicts.append(_evaluate_sentence(sentence, chunk_records, selected_method))
    result.nli_available = any(v.validation_method == "nli_entailment" for v in result.verdicts)

    claims = [v for v in result.verdicts if v.is_claim]
    result.claim_count = len(claims)
    counts = Counter(v.status for v in claims)
    result.supported_count = counts.get("supported", 0)
    result.weakly_supported_count = counts.get("weakly_supported", 0)
    result.unsupported_count = counts.get("unsupported", 0)
    result.claim_support_rate = (result.supported_count / result.claim_count) if result.claim_count else 1.0
    result.citation_status = _aggregate_citation_status(result)
    return result


def _split_sentences(text: str) -> list[str]:
    protected = (
        text.strip()
        .replace("St. John's", "St<dot> John's")
        .replace("St. John", "St<dot> John")
        .replace("e.g.", "e<dot>g<dot>")
        .replace("i.e.", "i<dot>e<dot>")
    )
    sentences = [s.replace("<dot>", ".").strip() for s in _SENT_SPLIT.split(protected) if s.strip()]
    return sentences


def _tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if t and t not in _STOPWORDS and len(t) > 1
    }


def _chunk_id(chunk: Mapping[str, object]) -> str:
    return str(chunk.get("id") or chunk.get("chunk_id") or chunk.get("parent_id") or "")


def _is_claim_sentence(sentence: str) -> bool:
    lower = sentence.lower()
    if any(signal in lower for signal in _CLAIM_SIGNALS):
        return True
    return bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:mg|ml|mcg|g|%|percent|cycle|cycles|day|days|week|weeks)\b", lower))


def _evaluate_sentence(sentence: str, chunk_records: list[dict[str, object]], method: str) -> ClaimVerdict:
    if not _is_claim_sentence(sentence):
        return ClaimVerdict(sentence=sentence, is_claim=False, support_score=0.0, status="non_claim", validation_method=method)

    sentence_toks = _tokens(sentence)
    if not sentence_toks:
        return ClaimVerdict(sentence=sentence, is_claim=True, support_score=0.0, status="unsupported", reason="no_substantive_tokens", validation_method=method)

    best_score = 0.0
    supporting: list[str] = []
    scored_chunks: list[tuple[float, str, str]] = []
    for chunk in chunk_records:
        chunk_id = str(chunk.get("id") or "")
        chunk_toks = chunk.get("tokens") or set()
        if not chunk_toks:
            continue
        overlap = len(sentence_toks & chunk_toks) / max(1, len(sentence_toks))
        scored_chunks.append((overlap, chunk_id, str(chunk.get("text") or "")))
        best_score = max(best_score, overlap)
        if overlap >= WEAKLY_SUPPORTED_THRESHOLD:
            supporting.append(chunk_id)

    status = "supported" if best_score >= SUPPORTED_THRESHOLD else "weakly_supported" if best_score >= WEAKLY_SUPPORTED_THRESHOLD else "unsupported"
    reason = None if status != "unsupported" else "no_chunk_above_weak_support_threshold"
    validation_method = "heuristic_overlap"
    entailment_score = None
    contradiction_score = None

    if method == "nli":
        nli = _evaluate_with_nli(sentence, scored_chunks)
        if nli["available"]:
            validation_method = "nli_entailment"
            entailment_score = float(nli["entailment_score"])
            contradiction_score = float(nli["contradiction_score"])
            if nli["supporting_chunk_ids"]:
                supporting = list(nli["supporting_chunk_ids"])
            best_score = max(best_score, entailment_score)
            if contradiction_score >= NLI_CONTRADICTION_THRESHOLD:
                status = "unsupported"
                reason = "nli_contradiction_detected"
            elif entailment_score >= NLI_ENTAILMENT_THRESHOLD:
                status = "supported"
                reason = None
            elif entailment_score >= 0.42:
                status = "weakly_supported"
                reason = None
            else:
                status = "unsupported"
                reason = "nli_no_entailing_chunk"
        else:
            validation_method = "heuristic_overlap_nli_unavailable"
            reason = reason or str(nli.get("reason") or "nli_model_unavailable")

    return ClaimVerdict(
        sentence=sentence,
        is_claim=True,
        support_score=best_score,
        status=status,
        supporting_chunk_ids=supporting,
        reason=reason,
        validation_method=validation_method,
        entailment_score=entailment_score,
        contradiction_score=contradiction_score,
    )


def _selected_method(method: str | None) -> str:
    selected = (method or os.getenv(VALIDATOR_ENV) or "heuristic").strip().lower()
    return "nli" if selected in {"nli", "entailment", "nli_entailment"} else "heuristic_overlap"


def _evaluate_with_nli(sentence: str, scored_chunks: list[tuple[float, str, str]]) -> dict[str, object]:
    nli = _load_nli_pipeline()
    if nli is None:
        return {"available": False, "reason": "nli_model_unavailable"}
    candidates = sorted(scored_chunks, reverse=True)[:3]
    best_entailment = 0.0
    best_contradiction = 0.0
    supporting: list[str] = []
    for _, chunk_id, text in candidates:
        if not text.strip():
            continue
        scores = _nli_scores(nli, premise=text[:1800], hypothesis=sentence[:500])
        entail = scores.get("entailment", 0.0)
        contra = scores.get("contradiction", 0.0)
        if entail > best_entailment:
            best_entailment = entail
            supporting = [chunk_id] if chunk_id else []
        best_contradiction = max(best_contradiction, contra)
    return {
        "available": True,
        "entailment_score": best_entailment,
        "contradiction_score": best_contradiction,
        "supporting_chunk_ids": supporting,
    }


def _nli_scores(nli, *, premise: str, hypothesis: str) -> dict[str, float]:
    output = nli({"text": premise, "text_pair": hypothesis}, truncation=True)
    if isinstance(output, list) and output and isinstance(output[0], list):
        output = output[0]
    labels: dict[str, float] = {}
    for item in output if isinstance(output, list) else [output]:
        label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0.0))
        if "entail" in label:
            labels["entailment"] = max(labels.get("entailment", 0.0), score)
        elif "contrad" in label:
            labels["contradiction"] = max(labels.get("contradiction", 0.0), score)
        elif "neutral" in label:
            labels["neutral"] = max(labels.get("neutral", 0.0), score)
    return labels


@lru_cache(maxsize=1)
def _load_nli_pipeline():
    try:
        from transformers import pipeline
    except Exception:
        return None
    model_name = os.getenv(NLI_MODEL_ENV, "typeform/distilbert-base-uncased-mnli")
    try:
        return pipeline("text-classification", model=model_name, top_k=None)
    except Exception:
        return None


def _aggregate_citation_status(result: ClaimValidationResult) -> str:
    if result.claim_count == 0:
        return "complete"
    if result.unsupported_count == 0 and result.weakly_supported_count == 0:
        return "complete"
    if result.unsupported_count == 0:
        return "partial"
    if result.supported_count == 0 and result.weakly_supported_count == 0:
        return "unsupported"
    return "partial" if result.unsupported_count <= result.claim_count // 2 else "unsupported"


__all__ = [
    "SUPPORTED_THRESHOLD",
    "WEAKLY_SUPPORTED_THRESHOLD",
    "NLI_ENTAILMENT_THRESHOLD",
    "NLI_CONTRADICTION_THRESHOLD",
    "ClaimValidationResult",
    "ClaimVerdict",
    "validate_claims",
]
