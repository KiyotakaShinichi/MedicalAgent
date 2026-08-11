"""Query-to-evidence adjudication for research-backed RAG answers.

Retrieval relevance is not the same thing as claim support.  This module
looks at the *requested claim* and the governed research chunks before answer
generation.  It can distinguish a paper that is merely related from evidence
that overlaps the question's actual subject, and it blocks local-product or
patient-specific conclusions that the literature corpus cannot establish.

The implementation is deliberately deterministic and conservative.  It is an
engineering abstention layer, not medical entailment and not clinical review.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9./+-]*", re.IGNORECASE)

RESEARCH_CUES = frozenset({
    "paper", "papers", "research", "study", "studies", "trial", "trials",
    "literature", "publication", "publications", "evidence", "source", "sources",
})

STOPWORDS = frozenset({
    "about", "after", "again", "also", "among", "because", "before", "being",
    "could", "does", "from", "have", "into", "itself", "might", "other",
    "should", "their", "there", "these", "they", "this", "those", "through",
    "using", "what", "when", "where", "which", "while", "with", "would",
    "paper", "papers", "research", "study", "studies", "trial", "trials",
    "literature", "publication", "publications", "evidence", "source", "sources",
    "tell", "show", "says", "said", "answer", "question", "please", "correct",
})

SAFE_LIMITATION_CUES = (
    "does not prove", "doesn't prove", "cannot prove", "can't prove",
    "not enough", "not sufficient", "limitations", "why is not", "why isn't",
    "without concluding", "without confirming", "not the same as",
)

LOCAL_PRODUCT_PATTERN = re.compile(
    r"\b(this|our|the)\s+(app|prototype|agent|assistant|system|platform|portal)\b"
    r".{0,80}\b(safe|validated|valid|effective|ready|benefit|works?|reliable)\b"
    r"|\b(safe|validated|valid|effective|ready|benefit|works?|reliable)\b"
    r".{0,80}\b(this|our|the)\s+(app|prototype|agent|assistant|system|platform|portal)\b",
    re.IGNORECASE,
)

PERSONAL_INFERENCE_PATTERN = re.compile(
    r"\b(my|me|i|this patient|the patient)\b.{0,100}"
    r"\b(confirm|prove|authorize|choose|calculate|predict|diagnose|reclassify|establish)\b"
    r"|\b(confirm|prove|authorize|choose|calculate|predict|diagnose|reclassify|establish)\b"
    r".{0,100}\b(my|me|this patient|the patient)\b",
    re.IGNORECASE,
)

UNSUPPORTED_AUTHORITY_PATTERNS = (
    re.compile(r"\b(vus|uncertain variant|variant of uncertain significance)\b.{0,80}\b(pathogenic|positive|inherited cancer|definite risk)\b", re.I),
    re.compile(r"\b(ca\s*15[- ]?3|ca\s*27(?:\.29)?|cea|tumou?r marker)\b.{0,80}\b(confirm|prove|recurrence|metastasis|progression)\b", re.I),
    re.compile(r"\b(herbal|supplement|natural cure|turmeric|cannabis|cbd)\b.{0,80}\b(replace|instead of|avoid)\b.{0,50}\b(treatment|therapy|chemo|chemotherapy)\b", re.I),
    re.compile(r"\b(exact|exactly|personal)\b.{0,50}\b(survival|months? remaining|how long.*live)\b", re.I),
    re.compile(r"\b(authori[sz]e|choose|calculate|set)\b.{0,60}\b(dose|dosage|treatment change|therapy change)\b", re.I),
)

ABSTENTION_REPLY = (
    "The retrieved papers may be related, but they do not establish the claim "
    "in your question. I can summarize what the sources actually studied or "
    "help you prepare a question for the appropriate care-team reviewer."
)


@dataclass(frozen=True)
class ResearchEvidenceAnswerability:
    status: str
    applies: bool
    requires_abstention: bool
    reason: str
    requested_claim_token_count: int
    matched_claim_token_count: int
    claim_token_coverage: float
    research_chunk_count: int
    related_paper_count: int
    safe_reply: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "applies": self.applies,
            "requires_abstention": self.requires_abstention,
            "reason": self.reason,
            "requested_claim_token_count": self.requested_claim_token_count,
            "matched_claim_token_count": self.matched_claim_token_count,
            "claim_token_coverage": round(self.claim_token_coverage, 4),
            "research_chunk_count": self.research_chunk_count,
            "related_paper_count": self.related_paper_count,
            "safe_reply": self.safe_reply,
            "clinical_validation": False,
            "support_check_is_medical_entailment": False,
        }


def assess_research_evidence_answerability(
    *,
    query: str,
    chunks: Sequence[Mapping[str, Any]],
    intent: str | None,
    safety: Mapping[str, Any] | None,
) -> ResearchEvidenceAnswerability:
    """Assess whether retrieved research can support the requested claim.

    High-risk requests remain owned by the safety boundary.  For low-risk
    research questions, local-product claims and patient-specific inferences
    abstain even when related literature was retrieved.  Other explicit
    research questions use a transparent lexical coverage check as a weak
    support proxy; the result is never described as entailment.
    """
    safety = safety or {}
    query_normalized = " ".join(TOKEN_RE.findall((query or "").lower()))
    research_chunks = [chunk for chunk in chunks if _is_research_chunk(chunk)]
    paper_ids = {
        str(chunk.get("pmcid") or chunk.get("source_url") or chunk.get("source_name") or chunk.get("id"))
        for chunk in research_chunks
    }
    if str(safety.get("level") or "low_risk") == "high_risk":
        return _verdict(
            status="handled_by_safety_boundary",
            applies=bool(research_chunks or _has_research_cue(query_normalized)),
            requires_abstention=False,
            reason="high-risk request remains controlled by deterministic safety routing",
            query_tokens=set(),
            matched=set(),
            research_chunks=research_chunks,
            paper_ids=paper_ids,
        )

    explicit_research_request = _has_research_cue(query_normalized)
    if not explicit_research_request and not research_chunks:
        return _verdict(
            status="not_applicable",
            applies=False,
            requires_abstention=False,
            reason="no explicit research request and no research-paper context",
            query_tokens=set(),
            matched=set(),
            research_chunks=research_chunks,
            paper_ids=paper_ids,
        )

    safe_limitation_question = any(cue in query_normalized for cue in SAFE_LIMITATION_CUES)
    authority_overreach = (
        LOCAL_PRODUCT_PATTERN.search(query_normalized) is not None
        or PERSONAL_INFERENCE_PATTERN.search(query_normalized) is not None
        or any(pattern.search(query_normalized) for pattern in UNSUPPORTED_AUTHORITY_PATTERNS)
    ) and not safe_limitation_question

    query_tokens = _claim_tokens(query_normalized)
    context_tokens = _context_tokens(research_chunks[:8])
    matched = query_tokens & context_tokens
    coverage = len(matched) / max(len(query_tokens), 1)

    # A bibliographic lookup is not a request to prove an arbitrary medical
    # claim. When the query identifies a paper by title or source identifier,
    # downstream claim and citation validators still limit the answer to the
    # retrieved passage.
    identified_paper = _identified_paper_lookup(query_normalized, research_chunks)

    if authority_overreach:
        return _verdict(
            status="related_paper_only" if research_chunks else "no_supporting_research_evidence",
            applies=True,
            requires_abstention=True,
            reason="related literature cannot establish a local-product or patient-specific authority claim",
            query_tokens=query_tokens,
            matched=matched,
            research_chunks=research_chunks,
            paper_ids=paper_ids,
            safe_reply=ABSTENTION_REPLY,
        )
    if explicit_research_request and not research_chunks:
        return _verdict(
            status="no_supporting_research_evidence",
            applies=True,
            requires_abstention=True,
            reason="the governed context contains no research-paper evidence",
            query_tokens=query_tokens,
            matched=matched,
            research_chunks=research_chunks,
            paper_ids=paper_ids,
            safe_reply=ABSTENTION_REPLY,
        )
    if explicit_research_request and identified_paper:
        return _verdict(
            status="identified_paper_summary_candidate",
            applies=True,
            requires_abstention=False,
            reason=(
                "the query identifies a governed research source; downstream claim and citation "
                "validation still limit the answer to the retrieved passage"
            ),
            query_tokens=query_tokens,
            matched=matched,
            research_chunks=research_chunks,
            paper_ids=paper_ids,
        )
    if explicit_research_request and (len(matched) < 2 or coverage < 0.45):
        return _verdict(
            status="related_paper_only",
            applies=True,
            requires_abstention=True,
            reason="retrieved papers are topically related but weakly overlap the requested claim",
            query_tokens=query_tokens,
            matched=matched,
            research_chunks=research_chunks,
            paper_ids=paper_ids,
            safe_reply=ABSTENTION_REPLY,
        )
    return _verdict(
        status="claim_support_candidate",
        applies=bool(explicit_research_request or research_chunks),
        requires_abstention=False,
        reason=(
            "research context overlaps the requested claim; downstream claim and citation validation still apply"
        ),
        query_tokens=query_tokens,
        matched=matched,
        research_chunks=research_chunks,
        paper_ids=paper_ids,
    )


def _verdict(
    *,
    status: str,
    applies: bool,
    requires_abstention: bool,
    reason: str,
    query_tokens: set[str],
    matched: set[str],
    research_chunks: Sequence[Mapping[str, Any]],
    paper_ids: set[str],
    safe_reply: str | None = None,
) -> ResearchEvidenceAnswerability:
    return ResearchEvidenceAnswerability(
        status=status,
        applies=applies,
        requires_abstention=requires_abstention,
        reason=reason,
        requested_claim_token_count=len(query_tokens),
        matched_claim_token_count=len(matched),
        claim_token_coverage=len(matched) / max(len(query_tokens), 1),
        research_chunk_count=len(research_chunks),
        related_paper_count=len(paper_ids),
        safe_reply=safe_reply,
    )


def _has_research_cue(text: str) -> bool:
    return bool(set(TOKEN_RE.findall(text)) & RESEARCH_CUES)


def _claim_tokens(text: str) -> set[str]:
    return {
        token
        for token in TOKEN_RE.findall(text)
        if len(token) >= 4 and token not in STOPWORDS and not token.isdigit()
    }


def _context_tokens(chunks: Sequence[Mapping[str, Any]]) -> set[str]:
    output: set[str] = set()
    for chunk in chunks:
        combined = " ".join(
            str(chunk.get(key) or "")
            for key in ("title", "source_name", "topic", "section", "text")
        ).lower()
        output.update(_claim_tokens(combined))
    return output


def _identified_paper_lookup(
    query_normalized: str,
    chunks: Sequence[Mapping[str, Any]],
) -> bool:
    lookup_cues = (
        "paper titled",
        "paper title",
        "study titled",
        "article titled",
        "publication titled",
        "paper called",
        "find the paper",
        "find the study",
    )
    has_lookup_cue = any(cue in query_normalized for cue in lookup_cues)
    query_tokens = _lookup_tokens(query_normalized)
    for chunk in chunks:
        identifiers = {
            str(chunk.get(key) or "").strip().lower()
            for key in ("pmcid", "pmid", "doi")
            if str(chunk.get(key) or "").strip()
        }
        if identifiers and identifiers & query_tokens:
            return True
        if not has_lookup_cue:
            continue
        title_tokens = _lookup_tokens(
            str(chunk.get("title") or chunk.get("source_name") or "").lower()
        )
        if len(title_tokens) < 3:
            continue
        if len(title_tokens & query_tokens) / len(title_tokens) >= 0.9:
            return True
    return False


def _lookup_tokens(text: str) -> set[str]:
    return {
        normalized
        for token in TOKEN_RE.findall(text)
        if len(normalized := token.strip("./+-")) >= 2
    }


def _is_research_chunk(chunk: Mapping[str, Any]) -> bool:
    trust = str(chunk.get("trust_level") or "").lower()
    url = str(chunk.get("source_url") or "").lower()
    return bool(
        trust == "research_paper"
        or chunk.get("pmcid")
        or "pmc.ncbi.nlm.nih.gov" in url
        or "pubmed.ncbi.nlm.nih.gov" in url
    )


__all__ = [
    "ABSTENTION_REPLY",
    "ResearchEvidenceAnswerability",
    "assess_research_evidence_answerability",
]
