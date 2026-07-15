"""Emotional distress detection for NLCare queries.

Classifies the *affective* signal in a patient query into one of:

  * none
  * fear
  * despair
  * anxiety
  * denial
  * crisis

…and selects a `response_mode` for the answer composer:

  * normal_education
  * empathetic_support_plus_education
  * urgent_clinician_review
  * crisis_support
  * clinician_review_with_warm_handoff

The vocabulary explicitly covers English + Taglish.  This module is
distinct from the safety-scope check: a query can be safety low-risk
and still need an empathetic response mode (e.g. "I'm so scared
about my next scan").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


# Distress categories ranked by escalation priority.  ``crisis`` always
# wins; ``denial`` is the lowest priority but still surfaces an
# empathetic response.
CRISIS_TERMS: tuple[str, ...] = (
    "suicidal", "kill myself", "end my life", "want to die",
    "don't want to live", "do not want to live", "no point living",
    "self harm", "self-harm", "hurt myself", "harm myself",
    # Taglish
    "gusto ko nang mamatay", "ayoko nang mabuhay",
    "ayoko na mabuhay", "magpapakamatay", "magpapakamatay na ako",
    "wala nang silbi buhay ko", "walang silbi ang buhay ko",
)

DESPAIR_TERMS: tuple[str, ...] = (
    "hopeless", "no hope", "give up", "giving up", "i give up",
    "what's the point", "no reason to keep going", "cannot go on",
    "can't go on", "can't keep going", "cannot keep going",
    "exhausted and done",
    # Taglish
    "wala na akong pag asa", "wala na akong pag-asa",
    "sumusuko na ako", "ayoko na", "hindi ko na kaya",
    "pagod na pagod na ako", "wala na akong lakas",
)

FEAR_TERMS: tuple[str, ...] = (
    "scared", "terrified", "afraid", "i'm so scared", "i am so scared",
    "frightened", "fear of", "petrified",
    # Taglish
    "takot", "natatakot", "kinakabahan", "kabado", "nanginginig ako",
)

ANXIETY_TERMS: tuple[str, ...] = (
    "anxious", "anxiety", "panic", "panicking", "panic attack",
    "can't sleep", "cannot sleep", "overthinking", "worried sick",
    "very worried", "extremely worried",
    # Taglish
    "kabado", "balisa", "hindi makatulog", "puyat ako sa kakaisip",
    "puro iniisip ko", "lagi ko inaalala",
)

DENIAL_TERMS: tuple[str, ...] = (
    "this isn't happening", "this is not happening", "i don't believe",
    "i do not believe", "can't be real", "cannot be real",
    "must be a mistake", "they got it wrong", "no way this is",
    # Taglish
    "hindi ako naniniwala", "siguro nagkamali sila",
    "imposible ito", "imposible 'to",
)


RESPONSE_MODE_VALUES: tuple[str, ...] = (
    "normal_education",
    "empathetic_support_plus_education",
    "urgent_clinician_review",
    "crisis_support",
    "clinician_review_with_warm_handoff",
)


@dataclass
class EmotionalDistressVerdict:
    detected: bool
    category: str
    response_mode: str
    matched_terms: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "category": self.category,
            "response_mode": self.response_mode,
            "matched_terms": list(self.matched_terms),
            "notes": self.notes,
        }


def _hits(text: str, terms: Sequence[str]) -> list[str]:
    lower = text.lower()
    return [t for t in terms if t in lower]


def detect_emotional_distress(
    query: str,
    safety: Mapping[str, Any] | None = None,
) -> EmotionalDistressVerdict:
    """Classify the affective signal in ``query``.

    Precedence: crisis > despair > fear > anxiety > denial > none.
    The chosen ``response_mode`` is:

    * crisis → ``crisis_support``
    * despair + safety high_risk → ``urgent_clinician_review``
    * despair otherwise → ``clinician_review_with_warm_handoff``
    * fear / anxiety / denial → ``empathetic_support_plus_education``
    * none → ``normal_education``
    """
    safety = safety or {}
    safety_level = str(safety.get("level") or "low_risk")

    crisis = _hits(query, CRISIS_TERMS)
    if crisis:
        return EmotionalDistressVerdict(
            detected=True,
            category="crisis",
            response_mode="crisis_support",
            matched_terms=crisis[:5],
            notes="Crisis wording detected; route to crisis resources and clinician immediately.",
        )

    despair = _hits(query, DESPAIR_TERMS)
    if despair:
        mode = "urgent_clinician_review" if safety_level == "high_risk" else "clinician_review_with_warm_handoff"
        return EmotionalDistressVerdict(
            detected=True,
            category="despair",
            response_mode=mode,
            matched_terms=despair[:5],
            notes="Despair wording detected; warm-handoff to clinician.",
        )

    fear = _hits(query, FEAR_TERMS)
    if fear:
        return EmotionalDistressVerdict(
            detected=True,
            category="fear",
            response_mode="empathetic_support_plus_education",
            matched_terms=fear[:5],
            notes="Fear wording detected; acknowledge before educating.",
        )

    anxiety = _hits(query, ANXIETY_TERMS)
    if anxiety:
        return EmotionalDistressVerdict(
            detected=True,
            category="anxiety",
            response_mode="empathetic_support_plus_education",
            matched_terms=anxiety[:5],
            notes="Anxiety wording detected; acknowledge before educating.",
        )

    denial = _hits(query, DENIAL_TERMS)
    if denial:
        return EmotionalDistressVerdict(
            detected=True,
            category="denial",
            response_mode="empathetic_support_plus_education",
            matched_terms=denial[:5],
            notes="Denial wording detected; acknowledge without confirming/denying diagnosis.",
        )

    return EmotionalDistressVerdict(
        detected=False,
        category="none",
        response_mode="normal_education",
        matched_terms=[],
        notes="No affective signal above threshold.",
    )


def vocabulary_manifest() -> dict[str, Any]:
    return {
        "categories": {
            "crisis": list(CRISIS_TERMS),
            "despair": list(DESPAIR_TERMS),
            "fear": list(FEAR_TERMS),
            "anxiety": list(ANXIETY_TERMS),
            "denial": list(DENIAL_TERMS),
        },
        "response_mode_values": list(RESPONSE_MODE_VALUES),
    }


__all__ = [
    "ANXIETY_TERMS",
    "CRISIS_TERMS",
    "DENIAL_TERMS",
    "DESPAIR_TERMS",
    "EmotionalDistressVerdict",
    "FEAR_TERMS",
    "RESPONSE_MODE_VALUES",
    "detect_emotional_distress",
    "vocabulary_manifest",
]
