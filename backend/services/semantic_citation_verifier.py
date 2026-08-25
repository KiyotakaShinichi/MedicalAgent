from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
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


#: Bumped whenever the case set changes meaningfully. Revision 1 was the original
#: seven cases; revision 2 attaches a case to every contradiction rule the module
#: declares and to each way a source can be disallowed.
SEMANTIC_CLAIM_EVALUATION_REVISION = 2


def case_set_fingerprint(cases: list[CitationCase]) -> str:
    """A deterministic digest of exactly what was evaluated.

    Order-independent and content-addressed, so the same suite always produces
    the same value and any edit to a claim, snippet, expectation, or source
    attribute produces a different one. This is what separates a new evaluation
    from a re-dated old one: the timestamp moves either way, the fingerprint only
    moves when the cases do.
    """
    payload = sorted(
        json.dumps(
            {
                "case_id": case.case_id,
                "claim": case.claim,
                "snippets": list(case.snippets),
                "expected": case.expected,
                "allowed_use": case.allowed_use,
                "source_tier": case.source_tier,
                "source_staleness": case.source_staleness,
                "has_citation": case.has_citation,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for case in cases
    )
    return hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()


def _implementation_fingerprint() -> str:
    """Digest of the verifier logic the cases were scored by.

    A case set can stay identical while the code that judges it changes; without
    this, two artifacts with the same case fingerprint could describe different
    behaviour.
    """
    source = Path(__file__).read_bytes()
    rules = json.dumps(HIGH_RISK_CONTRADICTIONS, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(source + rules).hexdigest()[:32]


def _git_commit_sha() -> str:
    """Follows the convention in backend/services/artifact_manifest.py."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _runtime_fingerprint() -> dict[str, Any]:
    lock = Path(__file__).resolve().parents[2] / "uv.lock"
    lock_digest = (
        hashlib.sha256(lock.read_bytes()).hexdigest()[:32] if lock.exists() else None
    )
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "uv_lock_sha256_prefix": lock_digest,
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
        # Provenance: a reader can tell whether a newer timestamp reflects a new
        # evaluation or the same suite re-dated.
        "evaluation_revision": SEMANTIC_CLAIM_EVALUATION_REVISION,
        "git_commit_sha": _git_commit_sha(),
        "case_set_sha256": case_set_fingerprint(eval_cases),
        "case_count": len(eval_cases),
        "implementation_fingerprint": _implementation_fingerprint(),
        "runtime": _runtime_fingerprint(),
        "runtime_mode": "offline_deterministic",
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
    # --- revision 2 -----------------------------------------------------------
    # Revision 1 exercised two of the six contradiction rules the module already
    # declares, and one of the three ways a source can be disallowed. The rules
    # below are not new policy: each is an entry in HIGH_RISK_CONTRADICTIONS or a
    # branch of `allowed_source` that had no case attached to it. Cases are
    # derived from those declarations, never from an observed failure.
    CitationCase(
        case_id="supplement_false_safety_contradiction",
        claim="St John's wort is safe to take with chemotherapy, so you can continue it.",
        snippets=[
            "St John's wort induces CYP3A4 and can reduce the plasma concentration of "
            "several chemotherapy agents; concurrent use is not recommended."
        ],
        expected="contradicted",
    ),
    CitationCase(
        case_id="treatment_change_advice_contradiction",
        claim="You should stop chemotherapy this cycle if you feel tired.",
        snippets=[
            "Fatigue is a common treatment effect. Any change to a chemotherapy schedule "
            "is decided by the treating oncology team."
        ],
        expected="contradicted",
    ),
    CitationCase(
        case_id="false_reassurance_contradiction",
        claim="There is no need to contact your care team about a fever during chemotherapy.",
        snippets=[
            "Fever during chemotherapy can indicate neutropenic sepsis and is treated as "
            "an emergency requiring immediate contact with the care team."
        ],
        expected="contradicted",
    ),
    CitationCase(
        case_id="progression_overclaim_contradiction",
        claim="This scan confirms progression of your disease.",
        snippets=[
            "Imaging findings are interpreted by a radiologist and the treating team "
            "alongside clinical context; a single report does not establish progression."
        ],
        expected="contradicted",
    ),
    CitationCase(
        case_id="blocked_allowed_use_source",
        claim="Standard neutropenia management includes growth-factor support in selected patients.",
        snippets=["Growth-factor support may be considered for selected patients at risk of febrile neutropenia."],
        expected="disallowed_source",
        allowed_use="blocked",
    ),
    CitationCase(
        case_id="untrusted_tier_source",
        claim="Standard neutropenia management includes growth-factor support in selected patients.",
        snippets=["Growth-factor support may be considered for selected patients at risk of febrile neutropenia."],
        expected="disallowed_source",
        source_tier="T4",
    ),
    CitationCase(
        case_id="expired_source_staleness",
        claim="Standard neutropenia management includes growth-factor support in selected patients.",
        snippets=["Growth-factor support may be considered for selected patients at risk of febrile neutropenia."],
        expected="disallowed_source",
        source_staleness="expired",
    ),
    CitationCase(
        case_id="overclaim_beyond_evidence",
        claim="Your specific five-year outcome can be predicted precisely from this blood count.",
        snippets=[
            "Complete blood counts are used to monitor treatment tolerance such as "
            "neutrophil, haemoglobin and platelet trends."
        ],
        expected="unsupported",
    ),
    CitationCase(
        case_id="supported_tier_two_education_source",
        claim="Neutrophil counts are monitored during chemotherapy to track treatment tolerance.",
        snippets=[
            "Neutrophil counts are monitored during chemotherapy to track treatment tolerance."
        ],
        expected="supported",
        source_tier="T2",
    ),
    CitationCase(
        case_id="supported_among_unrelated_snippets",
        claim="Platelet counts are monitored during chemotherapy to track bleeding risk.",
        snippets=[
            "Endocrine therapy adherence is assessed at follow-up visits.",
            "Platelet counts are monitored during chemotherapy to track bleeding risk.",
            "Imaging schedules vary by treatment plan.",
        ],
        expected="supported",
    ),
]
