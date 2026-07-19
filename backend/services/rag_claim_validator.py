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
_ATOMIC_SPLIT = re.compile(r"\s*(?:;|\n+\s*[-*â€¢]\s+|\n+\s*\d+[.)]\s+)\s*")
_NUMERIC_FACT = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|%|percent|x10\^?\d+/[a-z]+|k/[a-z]+|g/dl|u/ml|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)

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
    claim_type: str = "other_medical_claim"
    polarity: str = "positive"
    temporality: str = "general"
    population_scope: str = "general"
    numeric_facts: list[str] = field(default_factory=list)
    alignment_checks: dict[str, object] = field(default_factory=dict)

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
            "claim_type": self.claim_type,
            "polarity": self.polarity,
            "temporality": self.temporality,
            "population_scope": self.population_scope,
            "numeric_facts": list(self.numeric_facts),
            "alignment_checks": dict(self.alignment_checks),
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
        {
            "id": _chunk_id(c),
            "text": str(c.get("text") or ""),
            "tokens": _tokens(str(c.get("text") or "")),
            "source_type": str(c.get("source_type") or c.get("record_type") or ""),
            "is_patient_record": bool(c.get("is_patient_record")) or str(c.get("source_type") or "").lower() in {"patient_record", "timeline_record"},
        }
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
    atomic: list[str] = []
    for sentence in sentences:
        atomic.extend(part.strip() for part in _ATOMIC_SPLIT.split(sentence) if part.strip())
    return atomic


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
    claim_type = _claim_type(sentence)
    polarity = _polarity(sentence)
    temporality = _temporality(sentence)
    population_scope = _population_scope(sentence)
    numeric_facts = _numeric_facts(sentence)
    if not _is_claim_sentence(sentence):
        return ClaimVerdict(
            sentence=sentence, is_claim=False, support_score=0.0, status="non_claim", validation_method=method,
            claim_type=claim_type, polarity=polarity, temporality=temporality,
            population_scope=population_scope, numeric_facts=numeric_facts,
        )

    sentence_toks = _tokens(sentence)
    if not sentence_toks:
        return ClaimVerdict(
            sentence=sentence, is_claim=True, support_score=0.0, status="unsupported",
            reason="no_substantive_tokens", validation_method=method, claim_type=claim_type,
            polarity=polarity, temporality=temporality, population_scope=population_scope,
            numeric_facts=numeric_facts,
        )

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

    alignment_checks, alignment_reason = _alignment_checks(
        sentence,
        chunk_records,
        numeric_facts=numeric_facts,
        population_scope=population_scope,
    )
    if alignment_reason:
        status = "unsupported"
        reason = alignment_reason
        validation_method = "heuristic_structured_alignment"

    contradiction_reason = _heuristic_contradiction_reason(
        sentence,
        [text for _, _, text in scored_chunks],
    )
    if contradiction_reason:
        status = "unsupported"
        reason = contradiction_reason
        validation_method = "heuristic_overlap_contradiction"

    if method == "nli":
        nli = _evaluate_with_nli(sentence, scored_chunks)
        if nli["available"]:
            validation_method = "nli_entailment"
            entailment_score = float(nli["entailment_score"])
            contradiction_score = float(nli["contradiction_score"])
            if nli["supporting_chunk_ids"]:
                supporting = list(nli["supporting_chunk_ids"])
            best_score = max(best_score, entailment_score)
            if contradiction_reason or alignment_reason:
                # Safety-first: the local heuristic catches narrow medical
                # inversions that small generic MNLI models often
                # over-entail. NLI may upgrade paraphrased support, but it
                # must not override these explicit risk-boundary patterns.
                status = "unsupported"
                reason = contradiction_reason or alignment_reason
            elif contradiction_score >= NLI_CONTRADICTION_THRESHOLD:
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
        claim_type=claim_type,
        polarity=polarity,
        temporality=temporality,
        population_scope=population_scope,
        numeric_facts=numeric_facts,
        alignment_checks=alignment_checks,
    )


def _claim_type(sentence: str) -> str:
    lower = sentence.lower()
    if any(term in lower for term in ("dose", "dosage", "mg", "should stop", "should start", "treatment")):
        return "treatment_or_dose"
    if any(term in lower for term in ("survival", "prognosis", "life expectancy", "recurrence")):
        return "prognosis_or_outcome"
    if any(term in lower for term in ("brca", "vus", "variant", "genetic", "germline")):
        return "genetic"
    if any(term in lower for term in ("tumor marker", "ca 15", "ca 27", "cea")):
        return "tumor_marker"
    if any(term in lower for term in ("wbc", "anc", "hemoglobin", "platelet")):
        return "laboratory"
    if any(term in lower for term in ("mri", "ct ", "ultrasound", "mammogram", "scan")):
        return "imaging"
    return "other_medical_claim"


def _polarity(sentence: str) -> str:
    lower = sentence.lower()
    return "negative" if re.search(r"\b(no|not|never|cannot|can't|does not|do not|without)\b", lower) else "positive"


def _temporality(sentence: str) -> str:
    lower = sentence.lower()
    if re.search(r"\b(will|future|next|going to)\b", lower):
        return "future"
    if re.search(r"\b(was|were|previous|last|yesterday|ago)\b", lower):
        return "past"
    if re.search(r"\b(now|today|currently|current)\b", lower):
        return "current"
    return "general"


def _population_scope(sentence: str) -> str:
    lower = sentence.lower()
    if re.search(r"\b(you have|your (?:cancer|tumou?r|lab|result|scan|marker|risk|prognosis)|for you|in your case|my (?:cancer|tumou?r|lab|result|scan|marker|risk|prognosis))\b", lower):
        return "patient_specific"
    return "general"


def _numeric_facts(sentence: str) -> list[str]:
    return [f"{value} {unit.lower()}" for value, unit in _NUMERIC_FACT.findall(sentence)]


def _alignment_checks(
    sentence: str,
    chunk_records: list[dict[str, object]],
    *,
    numeric_facts: list[str],
    population_scope: str,
) -> tuple[dict[str, object], str | None]:
    evidence = "\n".join(str(chunk.get("text") or "") for chunk in chunk_records).lower()
    normalized_evidence = re.sub(r"\s+", " ", evidence)
    missing_numeric = [fact for fact in numeric_facts if fact.lower() not in normalized_evidence]
    patient_record_available = any(bool(chunk.get("is_patient_record")) for chunk in chunk_records)
    lower = sentence.lower()
    absolute_claim = any(term in lower for term in ("always", "never", "definitely", "guarantees", "proves", "certainly"))
    conditional_evidence = any(term in evidence for term in ("may", "might", "can", "depends", "context", "not by itself"))
    checks: dict[str, object] = {
        "numeric_alignment": "passed" if not missing_numeric else "failed",
        "missing_numeric_facts": missing_numeric,
        "patient_scope_alignment": "passed" if population_scope != "patient_specific" or patient_record_available else "failed",
        "patient_record_available": patient_record_available,
        "modality_alignment": "failed" if absolute_claim and conditional_evidence else "passed",
    }
    if missing_numeric:
        return checks, "numeric_value_or_unit_not_found_in_evidence"
    if population_scope == "patient_specific" and not patient_record_available:
        return checks, "patient_specific_claim_supported_only_by_generic_evidence"
    if absolute_claim and conditional_evidence:
        return checks, "absolute_claim_exceeds_conditional_evidence"
    return checks, None


def _heuristic_contradiction_reason(sentence: str, chunk_texts: list[str]) -> str | None:
    """Catch high-risk semantic inversions when the optional NLI model is
    unavailable.

    This is deliberately narrow. It is not a general entailment model; it
    covers the exact medical-safety inversions that token overlap gets wrong
    in CI: "safe/no review needed" against interaction guidance, tumor marker
    proof claims against limitation text, and treatment/dose recommendations
    against clinician-review boundaries.
    """
    lower = sentence.lower()
    evidence = "\n".join(chunk_texts).lower()

    says_safe_without_review = (
        any(phrase in lower for phrase in (
            "safe with",
            "is safe",
            "are safe",
            "no need",
            "does not need",
            "do not need",
            "without oncology review",
            "without review",
        ))
        and any(term in lower for term in (
            "st john", "johns wort", "john's wort", "supplement", "herbal",
            "cancer treatment", "chemotherapy", "chemo",
        ))
        and any(term in evidence for term in (
            "interact", "interaction", "discuss", "ask", "review",
            "oncology team", "pharmacist", "before use",
        ))
    )
    if says_safe_without_review:
        return "heuristic_contradiction_safety_review_required"

    says_marker_proves = (
        any(phrase in lower for phrase in (
            "proves",
            "confirms",
            "means cancer is back",
            "means recurrence",
            "diagnoses recurrence",
            "diagnose recurrence",
        ))
        and any(term in lower for term in ("ca 15-3", "ca 27.29", "cea", "tumor marker"))
        and any(term in evidence for term in (
            "cannot diagnose", "not standalone", "by itself", "context-dependent",
            "reviewed with symptoms", "reviewed with imaging",
        ))
    )
    if says_marker_proves:
        return "heuristic_contradiction_tumor_marker_overclaim"

    says_treatment_change = (
        any(phrase in lower for phrase in (
            "you should stop", "you should start", "you should change",
            "you should decrease", "you should increase", "recommended dose",
            "must stop", "must start", "must change",
        ))
        and any(term in lower for term in (
            "chemo", "chemotherapy", "dose", "tamoxifen", "paclitaxel",
            "trastuzumab", "doxorubicin",
        ))
        and any(term in evidence for term in (
            "clinician", "oncology team", "doctor", "review", "must not recommend",
        ))
    )
    if says_treatment_change:
        return "heuristic_contradiction_treatment_review_required"

    return None


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
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    except Exception:
        return None
    model_name = os.getenv(NLI_MODEL_ENV, "typeform/distilbert-base-uncased-mnli")
    allow_download = os.getenv("ONCOTRACK_NLI_ALLOW_DOWNLOAD", "").strip().lower() in {"1", "true", "yes"}
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=not allow_download)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=not allow_download)
        return pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)
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
