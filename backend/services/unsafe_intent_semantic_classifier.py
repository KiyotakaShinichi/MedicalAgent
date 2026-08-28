"""Generalized unsafe-intent classifier for patient-support routing.

This is a lightweight hybrid classifier: deterministic high-confidence
patterns first, then prototype-token similarity for paraphrases. It does not
replace the existing security guardrail, medical claim boundary checker, or
post-generation validator; it gives routing one more generalized signal before
generation.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.services.agent_text_normalization import normalize_agent_text
from backend.services.unsafe_intent_compositional_rules import (
    COMPOSITIONAL_RULES,
    DECISION_OR_ACCESS_CUES,
    SAFE_EDUCATIONAL_ANCHORS,
)
from backend.services.unsafe_intent_families import FAMILIES
from backend.services.unsafe_intent_safe_boundary import (
    looks_like_recording_statement as _looks_like_recording_statement,
    looks_like_safe_boundary_request as _looks_like_safe_boundary_request,
)


DEFAULT_OUTPUT_PATH = Path("Data/evals/safety/latest_unsafe_intent_classifier_eval.json")

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'/-]*", re.IGNORECASE)


def classify_unsafe_intent(text: str) -> dict[str, Any]:
    normalized = _normalize(text)
    # Security, safety, and bounded-workflow layers can consult this service
    # independently during one turn. Copy-on-read avoids recomputing an
    # identical normalized query without exposing a mutable cached result.
    return dict(_classify_normalized_cached(normalized))


@lru_cache(maxsize=2048)
def _classify_normalized_cached(normalized: str) -> dict[str, Any]:
    if not normalized:
        return _result("none", 0.0, "low_risk", "education_or_tracking", "none", False)
    if _looks_like_recording_statement(normalized):
        return _result(
            "none", 0.0, "low_risk", "education_or_tracking", "recording_statement", False
        )
    if _looks_like_safe_boundary_request(normalized):
        return _result(
            "none", 0.0, "low_risk", "education_or_tracking", "safe_boundary_request", False
        )
    if _looks_like_safe_education(normalized) and re.search(r"\bdose\s+dense\b", normalized):
        return _result("none", 0.0, "low_risk", "education_or_tracking", "none", False)

    best: dict[str, Any] | None = None
    best_selection_score = -1.0
    for family in FAMILIES:
        neg_score = _prototype_score(normalized, family.safe_negative_prototypes)
        pattern_match = _pattern_match(normalized, family.deterministic_patterns)
        compositional_score, compositional_rule = _compositional_match(normalized, family.family)
        pos_score = max(
            _prototype_score(normalized, family.positive_prototypes),
            _prototype_score(normalized, family.near_boundary_examples),
            _prototype_score(normalized, family.taglish_variants),
            compositional_score,
        )
        confidence = max(0.0, min(1.0, pos_score - (0.35 * neg_score)))
        source = "semantic_classifier"
        if pattern_match:
            confidence = max(confidence, 0.92)
            source = "deterministic"
        elif compositional_rule:
            confidence = max(confidence, compositional_score)
            source = "compositional_semantic"
        safe_education = _looks_like_safe_education(normalized)
        explicit_safe_framing = _has_explicit_protective_framing(normalized) or any(
            cue in normalized
            for cue in (
                "not automatically",
                "does not prove",
                "doesn't prove",
                "alone does not",
                "without concluding",
                "without deciding",
                "without diagnosing",
                "without reclassifying",
                "do not reveal",
                "don't reveal",
                "never reveal",
            )
        )
        safe_education = safe_education and (
            explicit_safe_framing or (pattern_match is None and compositional_rule is None)
        )
        if safe_education:
            confidence = min(confidence, 0.48)
        candidate = _result(
            family.family,
            confidence,
            family.expected_route,
            family.expected_scope,
            source,
            neg_score >= 0.45 or safe_education,
            template=family.safe_template,
            notes=family.over_refusal_risk_notes,
            matched_pattern=pattern_match,
            matched_semantic_rule=compositional_rule,
        )
        selection_score = candidate["confidence"] + _family_specificity_bonus(
            normalized, family.family
        )
        if best is None or selection_score > best_selection_score:
            best = candidate
            best_selection_score = selection_score

    if (
        best
        and best["confidence"] >= 0.62
        and not (best["over_refusal_risk_flag"] and best["confidence"] < 0.88)
    ):
        return best
    if best and best["confidence"] >= 0.50:
        return {**best, "route": "safe_clarification", "borderline": True}
    return _result(
        "none",
        best["confidence"] if best else 0.0,
        "low_risk",
        "education_or_tracking",
        "none",
        False,
    )


def evaluate_unsafe_intent_classifier(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    from backend.services.unsafe_intent_classifier_eval import evaluate_classifier

    return evaluate_classifier(
        output_path=output_path,
        families=FAMILIES,
        classify=classify_unsafe_intent,
    )


def _result(
    family: str,
    confidence: float,
    route: str,
    scope: str,
    source: str,
    over_refusal: bool,
    *,
    template: str | None = None,
    notes: str | None = None,
    matched_pattern: str | None = None,
    matched_semantic_rule: str | None = None,
) -> dict[str, Any]:
    return {
        "is_unsafe": family != "none",
        "family": family,
        "confidence": round(float(confidence), 4),
        "route": route,
        "scope": scope,
        "safety_source": source,
        "unsafe_intent_family": family,
        "unsafe_intent_confidence": round(float(confidence), 4),
        "over_refusal_risk_flag": bool(over_refusal),
        "safe_template": template,
        "over_refusal_risk_notes": notes,
        "matched_pattern": matched_pattern,
        "matched_semantic_rule": matched_semantic_rule,
    }


def _normalize(text: str) -> str:
    return normalize_agent_text(text)


def _tokens(text: str) -> set[str]:
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "with",
        "is",
        "are",
        "can",
        "i",
        "my",
        "me",
        "you",
        "this",
        "that",
        "what",
    }
    return {
        token.lower()
        for token in TOKEN_RE.findall(text)
        if token.lower() not in stop and len(token) > 2
    }


def _prototype_score(text: str, prototypes: tuple[str, ...]) -> float:
    if not prototypes:
        return 0.0
    text_tokens = _tokens(text)
    if not text_tokens:
        return 0.0
    best = 0.0
    for proto, proto_tokens in _normalized_prototypes(prototypes):
        if not proto_tokens:
            continue
        overlap = len(text_tokens & proto_tokens) / max(len(proto_tokens), 1)
        jaccard = len(text_tokens & proto_tokens) / max(len(text_tokens | proto_tokens), 1)
        substring = 0.25 if proto in text or text in proto else 0.0
        score = min(1.0, (0.72 * overlap) + (0.38 * jaccard) + substring)
        best = max(best, score)
    return best


@lru_cache(maxsize=None)
def _normalized_prototypes(prototypes: tuple[str, ...]) -> tuple[tuple[str, frozenset[str]], ...]:
    """Cache immutable prototype features reused by every classifier call."""

    return tuple(
        (normalized, frozenset(_tokens(normalized)))
        for prototype in prototypes
        if (normalized := _normalize(prototype))
    )


def _pattern_match(text: str, patterns: tuple[str, ...]) -> str | None:
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return pattern
    return None


def _compositional_match(text: str, family: str) -> tuple[float, str | None]:
    groups = COMPOSITIONAL_RULES.get(family, ())
    if not groups:
        return 0.0, None
    matched = []
    for group in groups:
        cue = next((term for term in group if _contains_cue(text, term)), None)
        if cue is None:
            return 0.0, None
        matched.append(cue)
    # Three independently matched concepts provide slightly more confidence
    # than a two-concept rule, but neither bypasses downstream validators.
    score = 0.94 if len(groups) >= 3 else 0.90
    return score, f"{family}:" + " + ".join(matched)


def _has_decision_or_access_cue(text: str) -> bool:
    return any(_contains_cue(text, cue) for cue in DECISION_OR_ACCESS_CUES)


def _contains_cue(text: str, cue: str) -> bool:
    """Match short concepts as tokens so ``atm`` cannot match ``treatment``."""

    if cue and cue[0].isalnum() and cue[-1].isalnum():
        return (
            re.search(
                rf"(?<![a-z0-9]){re.escape(cue)}(?![a-z0-9])",
                text,
                re.IGNORECASE,
            )
            is not None
        )
    return cue in text


def _looks_like_safe_education(text: str) -> bool:
    anchors = SAFE_EDUCATIONAL_ANCHORS + (
        "what questions should i ask",
        "what should i ask",
        "what questions could i",
        "questions could i",
        "why is",
        "why are",
        "why must",
        "how are",
        "how do",
        "can you help organize",
        "as a general concept",
    )
    if not any(anchor in text for anchor in anchors):
        return False
    protective_framing = (
        "not automatically",
        "does not prove",
        "doesn't prove",
        "alone does not",
        "without concluding",
        "without deciding",
        "without diagnosing",
        "should not be treated",
        "shouldn't be treated",
        "must not be treated",
        "questions could i bring",
        "questions should i bring",
        "questions can i bring",
        "for review",
    )
    if _has_explicit_protective_framing(text) or any(cue in text for cue in protective_framing):
        return True
    personal_or_decision = (
        "my result",
        "my report",
        "my risk",
        "for me",
        "from this",
        "based on this",
        "should i take",
        "should i stop",
        "should i change",
        "should i skip",
        "tell me whether",
    )
    return not any(cue in text for cue in personal_or_decision)


def _has_explicit_protective_framing(text: str) -> bool:
    """Recognize bounded negation that asks why a conclusion must not be made."""

    return (
        re.search(
            r"\b(?:should|must|can)\s+(?:[a-z0-9'-]+\s+){0,5}not\s+be\s+treated\s+as\b"
            r"|\b(?:shouldn't|mustn't|cannot|can't)\s+(?:[a-z0-9'-]+\s+){0,5}be\s+treated\s+as\b",
            text,
            re.IGNORECASE,
        )
        is not None
    )


def _family_specificity_bonus(text: str, family: str) -> float:
    """Prefer domain-specific boundaries over broad diagnosis/treatment ties."""

    if family == "cross_patient_exfiltration" and re.search(
        r"\b(another|other|different|unrelated|every|each|all|next|previous)\s+patient'?s?\b|\bpatient\s+(?=[a-z0-9 -]{0,30}\d{3,}\b)[a-z0-9][a-z0-9 -]{4,}\b|\b(relative|cousin|partner|spouse|neighbor|coworker)'?s?\b",
        text,
    ):
        return 0.18
    if family == "vus_misinterpretation" and re.search(
        r"\b(vus|variant of uncertain significance|uncertain variant|unclear variant|unclassified variant|unclassified brca|uncertain genetic|inconclusive genetic|inconclusive hereditary|indeterminate hereditary)\b",
        text,
    ):
        return 0.12
    if family == "tumor_marker_conclusion" and re.search(
        r"\b(ca\s*15(?:-|\s)?3|ca\s*27(?:\.|\s)?29|cea|tumou?r marker|rising marker|marker trend|marker)\b",
        text,
    ):
        return 0.12
    if family == "supplement_replacement" and re.search(
        r"\b(herbs?|herbal|supplements?|vitamins?|cannabis|turmeric|cbd|natural)\b",
        text,
    ):
        return 0.12
    return 0.0


__all__ = [
    "FAMILIES",
    "classify_unsafe_intent",
    "evaluate_unsafe_intent_classifier",
]
