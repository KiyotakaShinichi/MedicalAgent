"""Portfolio claim-safety guardrails.

Checks the wording used about this project - README, CV, recruiter and
senior-engineer phrasing - against a banned-phrase list. The banned phrases are
the ones that would turn an engineering prototype into an implied clinical
product: "clinically validated", "FDA", "diagnoses patients". The allowed list
is the honest equivalent of each.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PORTFOLIO_PATH = Path("Data/evals/governance/latest_portfolio_claim_safety_check.json")


BANNED_AFFIRMATIVE_PHRASES: tuple[str, ...] = (
    "clinically validated",
    "production healthcare ready",
    "patient benefit",
    "diagnostic system",
    "treatment recommender",
    "proven safe",
    "clinician-approved",
    "fhir compliant",
    "hospital interoperable",
    "fda approved",
    "fda cleared",
    "ce marked",
    "hipaa compliant",
    "real-world evidence",
)


ALLOWED_PHRASES: tuple[str, ...] = (
    "engineering prototype",
    "synthetic-only ML signals",
    "not clinically validated",
    "non-diagnostic",
    "monitor-only",
    "intended for clinician review",
    "source-governed retrieval",
    "claim-level citation validation",
    "release-gate-enforced",
    "in-sample only",
    "improvement not proven",
    "informational artifact only",
)


def build_portfolio_claim_safety_check() -> dict[str, Any]:
    samples = [
        {
            "audience": "linkedin_one_line",
            "safe_version": (
                "Built an engineering prototype of a safety-first breast cancer "
                "monitoring agent with source-governed RAG, claim-level citation "
                "validation, and release-gate-enforced negative-result reporting; "
                "synthetic-only data, not clinically validated."
            ),
            "unsafe_version": (
                "Built a clinically validated, production-ready AI doctor that "
                "diagnoses breast cancer using FHIR-compliant patient data."
            ),
            "why_unsafe": (
                "claims clinical validation, diagnosis authority, FHIR compliance, "
                "and production readiness — none of which are true"
            ),
        },
        {
            "audience": "recruiter_short",
            "safe_version": (
                "Designed and shipped a synthetic-data oncology monitoring agent: "
                "hybrid RAG, source-tier governance, adversarial safety bank with "
                "held-out generalisation reported honestly, and a 120-artifact "
                "release gate with explicit anti-overclaim tests."
            ),
            "unsafe_version": (
                "Built an AI cancer agent that improves patient outcomes and "
                "supports clinical decision-making in hospitals."
            ),
            "why_unsafe": (
                "asserts patient outcomes and clinical decision support without any "
                "real-data evidence or clinician sign-off"
            ),
        },
        {
            "audience": "senior_engineer_technical",
            "safe_version": (
                "RAG architecture with 5 intent-aware source-governed modes, "
                "hybrid dense+sparse RRF, query rewriting, claim-level citation "
                "validation (heuristic by default, NLI opt-in), uncertainty-aware "
                "answerability routing, per-turn trace with chain-of-thought "
                "deny-list, stage-wise oracle diagnostic.  Source-governed stack "
                "does not exceed BM25 on raw recall on the in-sample goldset; "
                "negative results documented; held-out v2 prepared but not "
                "completed."
            ),
            "unsafe_version": (
                "RAG architecture that outperforms baselines on retrieval and is "
                "clinically validated for oncology decision support."
            ),
            "why_unsafe": (
                "'outperforms baselines' is false on the frozen goldset; clinical "
                "validation has not happened"
            ),
        },
        {
            "audience": "readme_summary_paragraph",
            "safe_version": (
                "MedicalAgent is a safety-first, non-diagnostic breast cancer "
                "monitoring engineering prototype.  It combines source-governed "
                "dense/sparse RAG, claim-level citation validation, deterministic "
                "pre-generation safety gates, adversarial safety regression with "
                "held-out generalisation reporting, and release-gate-enforced "
                "negative-result publication.  All ML signals are synthetic and "
                "monitor-only.  No clinician sign-off, no IRB, no real patient "
                "data."
            ),
            "unsafe_version": (
                "MedicalAgent is a clinically validated breast cancer monitoring "
                "system used in hospitals to improve patient outcomes."
            ),
            "why_unsafe": (
                "every clause is unverifiable under current constraints"
            ),
        },
        {
            "audience": "cv_bullet",
            "safe_version": (
                "Engineering prototype of a non-diagnostic oncology monitoring "
                "agent on synthetic data; documented negative results (pruner "
                "regression, held-out adversarial gap, full-stack not exceeding "
                "BM25); test-locked anti-overclaim invariants."
            ),
            "unsafe_version": (
                "Shipped clinically validated AI for breast cancer diagnosis used "
                "by oncologists."
            ),
            "why_unsafe": (
                "no clinician engagement, no validation, no diagnostic authority"
            ),
        },
    ]

    return {
        "schema_version": "portfolio_claim_safety_check_v1",
        "status": "informational",
        "label": "portfolio_claim_safety_check",
        "clinical_validation": False,
        "claim_boundary": (
            "This artifact is wording guidance only.  It is not clinical "
            "validation, not clinician sign-off, not IRB approval, and not "
            "proof of patient benefit.  The samples below are templates the "
            "project owner can adapt while staying inside the project's hard "
            "constraints (synthetic-only, no clinician, no IRB, no real patient "
            "data)."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "banned_affirmative_phrases": list(BANNED_AFFIRMATIVE_PHRASES),
        "allowed_phrases": list(ALLOWED_PHRASES),
        "audience_samples": samples,
        "guidance": (
            "If a sentence about the project would be false to say in a courtroom "
            "or in front of a regulator, it must not be said in a CV, README, "
            "LinkedIn, or recruiter blurb either.  When in doubt, use 'engineering "
            "prototype' and 'synthetic-only'."
        ),
    }


def write_portfolio_claim_safety_check(path: Path = PORTFOLIO_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_portfolio_claim_safety_check(), indent=2), encoding="utf-8")
    return path


__all__ = [
    "ALLOWED_PHRASES",
    "BANNED_AFFIRMATIVE_PHRASES",
    "PORTFOLIO_PATH",
    "build_portfolio_claim_safety_check",
    "write_portfolio_claim_safety_check",
]
