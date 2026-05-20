from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_PATH = "Data/evals/rag/latest_semantic_citation_verification.json"

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-+./]*", re.IGNORECASE)
NEGATION_RE = re.compile(r"\b(not|no|never|cannot|can't|doesn't|do not|should not|without)\b", re.IGNORECASE)

HIGH_RISK_CONTRADICTIONS: tuple[tuple[str, str], ...] = (
    (r"st\.?\s*john'?s\s+wort.*\b(safe|fine|okay)\b.*\b(chemo|chemotherapy|treatment)\b", "supplement_false_safety"),
    (r"\bCA\s*15-?3\b.*\b(proves|confirms|means|shows)\b.*\b(recurrence|came back|progression)\b", "tumor_marker_overclaim"),
    (r"\bVUS\b.*\b(positive|pathogenic|definitely)\b", "vus_overclaim"),
    (r"\b(stop|skip|change|increase|decrease)\b.*\b(chemo|chemotherapy|dose|treatment)\b", "treatment_change_advice"),
    (r"\bno need to contact\b.*\b(doctor|oncologist|care team|clinician)\b", "false_reassurance"),
    (r"\b(confirms|definitely|proves)\b.*\b(progression|metastasis|recurrence)\b", "diagnosis_or_progression_overclaim"),
)


@dataclass(frozen=True)
class CitationCase:
    case_id: str
    claim: str
    snippets: list[str]
    expected: str
    allowed_use: str = "education"
    source_tier: str = "T1"
    source_staleness: str = "current"
    has_citation: bool = True


def verify_claim_against_sources(
    claim: str,
    snippets: list[str],
    *,
    allowed_use: str = "education",
    source_tier: str = "T1",
    source_staleness: str = "current",
    has_citation: bool = True,
    min_support_score: float = 0.42,
) -> dict[str, Any]:
    contradiction = _high_risk_contradiction(claim)
    allowed_source = (
        source_tier in {"T1", "T2", "T3"}
        and allowed_use not in {"clinician_only", "blocked"}
        and source_staleness not in {"stale", "expired", "unknown_stale"}
    )
    best_score = max((_support_score(claim, snippet) for snippet in snippets), default=0.0)
    nli_backend = _optional_nli_backend_status()
    unsupported = best_score < min_support_score
    verdict = "supported"
    if not has_citation:
        verdict = "missing_citation"
    elif not allowed_source:
        verdict = "disallowed_source"
    elif contradiction:
        verdict = "contradicted"
    elif unsupported:
        verdict = "unsupported"
    return {
        "claim": claim,
        "verdict": verdict,
        "support_score": round(best_score, 4),
        "allowed_source": allowed_source,
        "source_tier": source_tier,
        "allowed_use": allowed_use,
        "source_staleness": source_staleness,
        "has_citation": has_citation,
        "contradiction_rule": contradiction,
        "semantic_backend": nli_backend,
        "claim_boundary": (
            "Semantic citation verification is a lightweight engineering check using lexical overlap, "
            "sequence similarity, negation/contradiction heuristics, optional local NLI availability checks, "
            "and source-governance metadata. It is not a clinical reviewer."
        ),
    }


def run_semantic_citation_verification_eval(
    *,
    output_path: str = DEFAULT_OUTPUT_PATH,
    cases: list[CitationCase] | None = None,
) -> dict[str, Any]:
    eval_cases = cases or DEFAULT_CASES
    results = []
    for case in eval_cases:
        result = verify_claim_against_sources(
            case.claim,
            case.snippets,
            allowed_use=case.allowed_use,
            source_tier=case.source_tier,
            source_staleness=case.source_staleness,
            has_citation=case.has_citation,
        )
        result["case_id"] = case.case_id
        result["expected"] = case.expected
        result["passed"] = result["verdict"] == case.expected
        results.append(result)
    hard_failures = [item for item in results if not item["passed"]]
    payload = {
        "schema_version": "semantic_citation_verification_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if not hard_failures else "needs_attention",
        "summary": {
            "case_count": len(results),
            "hard_failures": len(hard_failures),
            "supported_cases": sum(1 for item in results if item["verdict"] == "supported"),
            "unsupported_cases": sum(1 for item in results if item["verdict"] == "unsupported"),
            "contradicted_cases": sum(1 for item in results if item["verdict"] == "contradicted"),
            "disallowed_source_cases": sum(1 for item in results if item["verdict"] == "disallowed_source"),
            "missing_citation_cases": sum(1 for item in results if item["verdict"] == "missing_citation"),
            "nli_available_cases": sum(1 for item in results if item.get("semantic_backend", {}).get("nli_available")),
        },
        "cases": results,
        "claim_boundary": "Engineering citation support check only; not clinical validation or medical fact adjudication.",
    }
    _write_json(Path(output_path), payload)
    return payload


def run_semantic_claim_validation_eval(
    output_path: str = "Data/evals/rag/latest_semantic_claim_validation.json",
) -> dict[str, Any]:
    return run_semantic_citation_verification_eval(output_path=output_path, cases=SEMANTIC_CLAIM_CASES)


def extract_medical_claims(answer: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", answer.strip())
    medical_terms = re.compile(
        r"\b(cancer|chemo|chemotherapy|dose|treatment|HER2|ER|PR|VUS|BRCA|CA\s*15-?3|tumou?r marker|"
        r"recurrence|progression|doctor|oncologist|supplement|St\.?\s*John)\b",
        re.IGNORECASE,
    )
    return [part.strip() for part in parts if part.strip() and medical_terms.search(part)]


def _support_score(claim: str, snippet: str) -> float:
    claim_tokens = _important_tokens(claim)
    snippet_tokens = _important_tokens(snippet)
    if not claim_tokens:
        return 0.0
    overlap = len(claim_tokens & snippet_tokens) / len(claim_tokens)
    sequence = SequenceMatcher(None, " ".join(sorted(claim_tokens)), " ".join(sorted(snippet_tokens))).ratio()
    negation_penalty = 0.25 if bool(NEGATION_RE.search(claim)) != bool(NEGATION_RE.search(snippet)) else 0.0
    return max(0.0, min(1.0, max(overlap, sequence * 0.8) - negation_penalty))


def _important_tokens(text: str) -> set[str]:
    stop = {"the", "a", "an", "and", "or", "to", "of", "with", "is", "are", "can", "may", "your", "you"}
    return {token.lower() for token in TOKEN_RE.findall(text or "") if token.lower() not in stop and len(token) > 2}


def _high_risk_contradiction(claim: str) -> str | None:
    for pattern, rule in HIGH_RISK_CONTRADICTIONS:
        if re.search(pattern, claim or "", re.IGNORECASE | re.DOTALL):
            return rule
    return None


def _optional_nli_backend_status() -> dict[str, Any]:
    try:
        import transformers  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on local env
        return {
            "nli_available": False,
            "backend": "heuristic_similarity_fallback",
            "reason": f"transformers_unavailable:{exc.__class__.__name__}",
        }
    return {
        "nli_available": True,
        "backend": "local_transformers_available_optional",
        "reason": "NLI model loading is optional; deterministic contradiction rules remain mandatory.",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


DEFAULT_CASES: list[CitationCase] = [
    CitationCase(
        case_id="supported_cbc",
        claim="Low white blood cell counts during chemotherapy can increase infection risk and should be reviewed with the care team.",
        snippets=["Chemotherapy can lower white blood cell counts. Low counts may increase infection risk and should be discussed with the oncology care team."],
        expected="supported",
    ),
    CitationCase(
        case_id="unsupported_survival_number",
        claim="This pattern gives a 92 percent chance of cure.",
        snippets=["The source explains treatment monitoring and does not provide individual cure probabilities."],
        expected="unsupported",
    ),
    CitationCase(
        case_id="contradicted_st_johns_wort",
        claim="St. John's wort is safe with chemotherapy and no review is needed.",
        snippets=["Patients should tell their care team about supplements. St. John's wort can interact with some medicines."],
        expected="contradicted",
    ),
    CitationCase(
        case_id="contradicted_tumor_marker",
        claim="CA 15-3 proves recurrence.",
        snippets=["Tumor markers such as CA 15-3 are not used alone to diagnose recurrence."],
        expected="contradicted",
    ),
    CitationCase(
        case_id="missing_citation",
        claim="CBC trends can be monitored during chemotherapy.",
        snippets=["CBC trends can be monitored during treatment."],
        expected="missing_citation",
        has_citation=False,
    ),
    CitationCase(
        case_id="stale_source",
        claim="CBC trends can be monitored during chemotherapy.",
        snippets=["CBC trends can be monitored during treatment."],
        expected="disallowed_source",
        source_staleness="stale",
    ),
    CitationCase(
        case_id="disallowed_clinician_source",
        claim="CBC trends can be monitored during treatment.",
        snippets=["Clinician-only protocol text about CBC monitoring."],
        expected="disallowed_source",
        allowed_use="clinician_only",
        source_tier="T2",
    ),
]


SEMANTIC_CLAIM_CASES: list[CitationCase] = [
    CitationCase(
        case_id="supported_general_her2",
        claim="HER2 is a protein that can affect breast cancer treatment planning.",
        snippets=["HER2 is a protein. HER2 status helps clinicians plan breast cancer treatment."],
        expected="supported",
    ),
    CitationCase(
        case_id="unsupported_survival_estimate",
        claim="This gives a 70 percent five-year survival estimate.",
        snippets=["This source explains general education and does not provide personal survival estimates."],
        expected="unsupported",
    ),
    CitationCase(
        case_id="high_overlap_vus_contradiction",
        claim="A BRCA VUS means positive.",
        snippets=["A BRCA VUS does not mean positive or pathogenic; it requires genetics-trained review."],
        expected="contradicted",
    ),
    CitationCase(
        case_id="high_overlap_tumor_marker_contradiction",
        claim="CA 15-3 proves recurrence.",
        snippets=["CA 15-3 does not prove recurrence and should not be used alone to diagnose progression."],
        expected="contradicted",
    ),
    CitationCase(
        case_id="missing_citation_patient_education",
        claim="Low WBC can increase infection risk.",
        snippets=["Low WBC can increase infection risk."],
        expected="missing_citation",
        has_citation=False,
    ),
    CitationCase(
        case_id="disallowed_clinician_only",
        claim="CBC trends can be reviewed during treatment.",
        snippets=["Clinician-only CBC protocol."],
        expected="disallowed_source",
        allowed_use="clinician_only",
        source_tier="T2",
    ),
    CitationCase(
        case_id="stale_patient_source",
        claim="CBC trends can be reviewed during treatment.",
        snippets=["CBC trends can be reviewed during treatment."],
        expected="disallowed_source",
        source_staleness="stale",
    ),
]
