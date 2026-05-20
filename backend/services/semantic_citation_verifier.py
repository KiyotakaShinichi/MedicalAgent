from __future__ import annotations

import json
import re
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


def verify_claim_against_sources(
    claim: str,
    snippets: list[str],
    *,
    allowed_use: str = "education",
    source_tier: str = "T1",
    min_support_score: float = 0.42,
) -> dict[str, Any]:
    contradiction = _high_risk_contradiction(claim)
    allowed_source = source_tier in {"T1", "T2", "T3"} and allowed_use not in {"clinician_only", "blocked"}
    best_score = max((_support_score(claim, snippet) for snippet in snippets), default=0.0)
    unsupported = best_score < min_support_score
    verdict = "supported"
    if not allowed_source:
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
        "contradiction_rule": contradiction,
        "claim_boundary": (
            "Semantic citation verification is a lightweight engineering check using lexical overlap, "
            "negation/contradiction heuristics, and source-governance metadata. It is not a clinical reviewer."
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
        },
        "cases": results,
        "claim_boundary": "Engineering citation support check only; not clinical validation or medical fact adjudication.",
    }
    _write_json(Path(output_path), payload)
    return payload


def _support_score(claim: str, snippet: str) -> float:
    claim_tokens = _important_tokens(claim)
    snippet_tokens = _important_tokens(snippet)
    if not claim_tokens:
        return 0.0
    overlap = len(claim_tokens & snippet_tokens) / len(claim_tokens)
    negation_penalty = 0.25 if bool(NEGATION_RE.search(claim)) != bool(NEGATION_RE.search(snippet)) else 0.0
    return max(0.0, min(1.0, overlap - negation_penalty))


def _important_tokens(text: str) -> set[str]:
    stop = {"the", "a", "an", "and", "or", "to", "of", "with", "is", "are", "can", "may", "your", "you"}
    return {token.lower() for token in TOKEN_RE.findall(text or "") if token.lower() not in stop and len(token) > 2}


def _high_risk_contradiction(claim: str) -> str | None:
    for pattern, rule in HIGH_RISK_CONTRADICTIONS:
        if re.search(pattern, claim or "", re.IGNORECASE | re.DOTALL):
            return rule
    return None


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
        case_id="disallowed_clinician_source",
        claim="CBC trends can be monitored during treatment.",
        snippets=["Clinician-only protocol text about CBC monitoring."],
        expected="disallowed_source",
        allowed_use="clinician_only",
        source_tier="T2",
    ),
]
