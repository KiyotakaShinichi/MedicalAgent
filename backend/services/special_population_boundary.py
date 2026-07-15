"""Special-population boundary handling for NLCare.

Detects queries that touch pregnancy, breastfeeding, pediatric / minor,
fertility, survivorship, recurrence anxiety, palliative / supportive
care, and end-of-life distress.  These categories all need:

  - safe education only (no diagnosis / treatment / prognosis)
  - explicit clinician routing for any patient-specific advice
  - compassionate wording
  - urgent escalation for red flags (suicidality, self-harm,
    acute distress)

The detector is intentionally heuristic.  When it fires, the caller
should append the appropriate ``safe_wording`` template to (or
substitute it for) the agent reply.  The post-generation validator
remains authoritative — this module does not bypass it; it provides a
*recommended phrasing* surface for the answer composition layer.

Module contract
~~~~~~~~~~~~~~~

    classify_special_population(query: str) -> SpecialPopulationVerdict

``SpecialPopulationVerdict.category`` is one of:

  - ``"pregnancy"``
  - ``"breastfeeding"``
  - ``"pediatric"``
  - ``"fertility"``
  - ``"survivorship"``
  - ``"recurrence_anxiety"``
  - ``"palliative_or_supportive"``
  - ``"end_of_life_distress"``
  - ``None`` (no match)

When ``urgent_escalation`` is True, the caller MUST route to the urgent
safety lane (emergency / crisis resources).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ─── Trigger vocabularies ────────────────────────────────────────────────────


PREGNANCY_TERMS: tuple[str, ...] = (
    "pregnant", "pregnancy", "expecting", "trying to conceive",
    "first trimester", "second trimester", "third trimester",
    "buntis", "nagdadalang-tao",
)

BREASTFEEDING_TERMS: tuple[str, ...] = (
    "breastfeeding", "nursing my baby", "lactation", "pumping milk",
    "nagpapasuso",
)

PEDIATRIC_TERMS: tuple[str, ...] = (
    "my child", "my daughter", "my son", "12 year old", "13 year old",
    "14 year old", "15 year old", "16 year old", "17 year old",
    "minor patient", "teenager", "pediatric",
)

FERTILITY_TERMS: tuple[str, ...] = (
    "fertility", "egg freezing", "sperm banking", "ovarian preservation",
    "ivf", "iui", "have children later", "having kids after",
    "trying to have a baby",
)

SURVIVORSHIP_TERMS: tuple[str, ...] = (
    "survivorship", "after treatment finishes", "post treatment",
    "post-treatment", "i am done with chemo", "finished chemo",
    "remission", "survivor",
)

RECURRENCE_ANXIETY_TERMS: tuple[str, ...] = (
    "is it back", "did it come back", "scared of recurrence",
    "afraid of recurrence", "recurrence anxiety", "worried it returned",
    "worried the cancer came back", "scared the cancer came back",
    "bumalik ba",
)

PALLIATIVE_TERMS: tuple[str, ...] = (
    "palliative care", "supportive care for symptoms",
    "comfort care", "hospice", "managing my symptoms only",
)

END_OF_LIFE_DISTRESS_TERMS: tuple[str, ...] = (
    "don't want to live", "do not want to live", "want to end it",
    "thinking of suicide", "kill myself", "self harm",
    "ayoko nang mabuhay", "gusto ko nang mamatay",
)


# Categories that must always force an urgent_escalation path regardless
# of the rest of the response.
_URGENT_CATEGORIES: frozenset[str] = frozenset({
    "end_of_life_distress",
})


# ─── Safe-wording templates ──────────────────────────────────────────────────


SAFE_WORDING: dict[str, str] = {
    "pregnancy": (
        "Cancer care during pregnancy is highly individualized — choices about "
        "treatment, imaging, and monitoring depend on trimester, the regimen, "
        "and the obstetrics team. I cannot recommend or change any treatment "
        "or test in this situation. Please discuss this with your oncology "
        "team and your obstetrician together."
    ),
    "breastfeeding": (
        "Many cancer treatments can affect breastfeeding safety, and the "
        "specifics depend on the regimen and the timing of doses. I cannot "
        "say whether continuing or pausing breastfeeding is safe for your "
        "case. Please talk with your oncology team and your pediatrician / "
        "lactation consultant before making changes."
    ),
    "pediatric": (
        "NLCare is designed for adult breast-cancer monitoring contexts. "
        "Pediatric / minor patient care involves different specialists, "
        "consent requirements, and guidelines. Please contact a pediatric "
        "oncology team for any pediatric-specific question."
    ),
    "fertility": (
        "Fertility decisions during or after cancer treatment are time-"
        "sensitive and depend on the regimen, age, and reproductive goals. "
        "I cannot estimate fertility outcomes or recommend a path. Please "
        "ask your oncology team for a referral to a reproductive specialist "
        "or fertility preservation program."
    ),
    "survivorship": (
        "Survivorship planning is an active clinical area — surveillance "
        "imaging, long-term symptom management, lifestyle, and bone / heart "
        "health are all part of it. I can help you organize questions for a "
        "survivorship visit, but I cannot tell you what surveillance schedule "
        "applies to your case. That comes from your oncology team."
    ),
    "recurrence_anxiety": (
        "Worrying that cancer has returned is one of the most common "
        "experiences after treatment, and the feeling itself is not evidence "
        "either way. I cannot tell you whether cancer has come back — that "
        "is a clinical decision based on exam, imaging, and labs. Please "
        "share specific symptoms with your oncology team so they can advise "
        "on timing of the next visit or imaging."
    ),
    "palliative_or_supportive": (
        "Palliative and supportive care focuses on managing symptoms and "
        "quality of life and can be offered alongside any cancer treatment. "
        "I can help you organize the symptoms you want to discuss and "
        "questions to bring up, but the care plan itself is a clinical "
        "decision made with your palliative care or oncology team."
    ),
    "end_of_life_distress": (
        "I'm really sorry you are feeling this way. You deserve real "
        "support from a person right now. If you are in immediate danger or "
        "feel you might hurt yourself, please contact local emergency "
        "services or a crisis hotline now. You can also reach out to your "
        "oncology care team, who can help connect you with mental health "
        "support."
    ),
}


@dataclass
class SpecialPopulationVerdict:
    category: str | None = None
    urgent_escalation: bool = False
    matched_terms: list[str] = field(default_factory=list)
    safe_wording: str = ""
    routing: str = "oncology_care_team"

    def to_dict(self) -> dict[str, Any]:
        return {
            "category":          self.category,
            "urgent_escalation": self.urgent_escalation,
            "matched_terms":     list(self.matched_terms),
            "safe_wording":      self.safe_wording,
            "routing":           self.routing,
        }


_CATEGORY_TABLES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    # Order matters — end_of_life_distress must beat everything else.
    ("end_of_life_distress",     END_OF_LIFE_DISTRESS_TERMS, "crisis_resources_and_oncology"),
    ("pediatric",                PEDIATRIC_TERMS,            "pediatric_oncology_team"),
    ("pregnancy",                PREGNANCY_TERMS,            "oncology_and_obstetrics"),
    ("breastfeeding",            BREASTFEEDING_TERMS,        "oncology_and_pediatrics_or_lactation"),
    ("fertility",                FERTILITY_TERMS,            "oncology_and_reproductive_specialist"),
    ("palliative_or_supportive", PALLIATIVE_TERMS,           "palliative_care_team_or_oncology"),
    ("survivorship",             SURVIVORSHIP_TERMS,         "survivorship_clinic_or_oncology"),
    ("recurrence_anxiety",       RECURRENCE_ANXIETY_TERMS,   "oncology_care_team"),
)


def classify_special_population(query: str) -> SpecialPopulationVerdict:
    """Return the verdict for ``query``.  Empty input -> no category."""
    lower = (query or "").lower()
    if not lower.strip():
        return SpecialPopulationVerdict()

    for category, terms, routing in _CATEGORY_TABLES:
        matched = [term for term in terms if term in lower]
        if matched:
            return SpecialPopulationVerdict(
                category=category,
                urgent_escalation=category in _URGENT_CATEGORIES,
                matched_terms=matched,
                safe_wording=SAFE_WORDING.get(category, ""),
                routing=routing,
            )
    return SpecialPopulationVerdict()


def safe_wording_for(category: str) -> str:
    """Helper used by the answer composition layer."""
    return SAFE_WORDING.get(category, "")


__all__ = [
    "PREGNANCY_TERMS",
    "BREASTFEEDING_TERMS",
    "PEDIATRIC_TERMS",
    "FERTILITY_TERMS",
    "SURVIVORSHIP_TERMS",
    "RECURRENCE_ANXIETY_TERMS",
    "PALLIATIVE_TERMS",
    "END_OF_LIFE_DISTRESS_TERMS",
    "SAFE_WORDING",
    "SpecialPopulationVerdict",
    "classify_special_population",
    "safe_wording_for",
]
