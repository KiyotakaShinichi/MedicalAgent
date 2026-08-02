"""Conservative claim-to-source citation selection.

This module is intentionally separate from retrieval. It accepts already
governed chunks and chooses a small citation set for each generated claim.
The score is an engineering support proxy, not medical entailment. Live use
must remain disabled until paired generated-answer evaluation proves a gain.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "with", "you", "your",
}
DISALLOWED_USES = {"clinician_only", "internal_only", "not_patient_facing"}
TIER_BONUS = {"T1": 1.0, "T2": 0.8, "T3": 0.5, "T4": 0.1, "T5": 0.0}


def select_citations_for_claims(
    claims: Iterable[str],
    chunks: Iterable[Mapping[str, Any]],
    *,
    max_per_claim: int = 2,
    minimum_support_proxy: float = 0.34,
    refusal_route: bool = False,
) -> dict[str, Any]:
    """Assign governed chunks to claims without changing retrieval ranking."""

    claim_list = [str(claim).strip() for claim in claims if str(claim).strip()]
    chunk_list = [dict(chunk) for chunk in chunks]
    if refusal_route:
        return _result(claim_list, [], reason="refusal_route_strips_citations")

    eligible = [chunk for chunk in chunk_list if _eligible(chunk)]
    retrieval_scores = [_number(chunk.get("retrieval_score")) for chunk in eligible]
    low = min(retrieval_scores, default=0.0)
    high = max(retrieval_scores, default=0.0)
    assignments: list[dict[str, Any]] = []

    for claim in claim_list:
        claim_tokens = _tokens(claim)
        candidates = []
        for chunk in eligible:
            source_id = _source_id(chunk)
            chunk_tokens = _tokens(_chunk_text(chunk))
            overlap = claim_tokens & chunk_tokens
            coverage = len(overlap) / max(len(claim_tokens), 1)
            retrieval = _normalized(_number(chunk.get("retrieval_score")), low, high)
            tier = TIER_BONUS.get(str(chunk.get("source_tier") or "").upper(), 0.0)
            score = (0.72 * coverage) + (0.18 * retrieval) + (0.10 * tier)
            if len(overlap) < 2 and coverage < 0.25:
                continue
            candidates.append({
                "source_id": source_id,
                "support_proxy": round(score, 4),
                "lexical_coverage": round(coverage, 4),
                "matched_terms": sorted(overlap),
                "source_tier": chunk.get("source_tier"),
            })
        selected = [
            candidate
            for candidate in sorted(
                candidates,
                key=lambda row: (-row["support_proxy"], -row["lexical_coverage"], row["source_id"]),
            )
            if candidate["support_proxy"] >= minimum_support_proxy
        ][:max(1, int(max_per_claim))]
        assignments.append({
            "claim": claim,
            "supported_by_proxy": bool(selected),
            "selected_sources": selected,
        })

    selected_ids = []
    seen = set()
    for assignment in assignments:
        for source in assignment["selected_sources"]:
            if source["source_id"] and source["source_id"] not in seen:
                seen.add(source["source_id"])
                selected_ids.append(source["source_id"])
    return _result(claim_list, assignments, selected_ids=selected_ids)


def _result(
    claims: list[str],
    assignments: list[dict[str, Any]],
    *,
    selected_ids: list[str] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    unsupported = [row["claim"] for row in assignments if not row["supported_by_proxy"]]
    return {
        "selected_citation_ids": selected_ids or [],
        "claim_assignments": assignments,
        "claim_count": len(claims),
        "supported_claim_count": len(claims) - len(unsupported),
        "unsupported_claims": unsupported,
        "all_claims_supported_by_proxy": bool(claims) and not unsupported,
        "reason": reason,
        "live_patient_route_changed": False,
        "clinical_validation": False,
        "support_proxy_is_entailment": False,
    }


def _eligible(chunk: Mapping[str, Any]) -> bool:
    if bool(chunk.get("stale")) or bool(chunk.get("is_stale")):
        return False
    allowed_use = str(chunk.get("allowed_use") or "").lower()
    return allowed_use not in DISALLOWED_USES


def _tokens(value: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(value)
        if token.lower() not in STOP_WORDS and len(token) > 2
    }


def _chunk_text(chunk: Mapping[str, Any]) -> str:
    return " ".join(
        str(chunk.get(key) or "")
        for key in ("title", "topic", "text", "content", "snippet", "source_name")
    )


def _source_id(chunk: Mapping[str, Any]) -> str:
    return str(
        chunk.get("parent_id")
        or chunk.get("source_id")
        or chunk.get("id")
        or chunk.get("chunk_id")
        or ""
    ).strip()


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalized(value: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0 if value > 0 else 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


__all__ = ["select_citations_for_claims"]
