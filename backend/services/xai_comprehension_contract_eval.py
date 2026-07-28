"""Internal proxy evaluation for patient-visible explanation contracts.

This is a wording and information-completeness check, not a user study. It
cannot establish comprehension, usability, informed consent, or clinical
safety without independent human participants and appropriate oversight.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OUTPUT_PATH = Path("Data/evals/models/latest_xai_comprehension_contract_eval.json")
AUTHORITY_PATTERNS = (
    r"\byour cancer (?:is|has|will)\b",
    r"\btreatment (?:will|is guaranteed to) work\b",
    r"\b(?:start|stop|increase|decrease|switch) (?:your )?(?:dose|medication|treatment|chemotherapy)\b",
    r"\byou have \d+ (?:days|weeks|months|years) to live\b",
    r"\bdoctor (?:was|has been) notified\b",
    r"\bclinically validated\b",
)


def _syllables(word: str) -> int:
    groups = re.findall(r"[aeiouy]+", word.lower())
    return max(1, len(groups))


def _flesch_reading_ease(text: str) -> float:
    words = re.findall(r"[A-Za-z]+", text)
    sentences = [part for part in re.split(r"[.!?]+", text) if part.strip()]
    if not words or not sentences:
        return 0.0
    score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (sum(_syllables(word) for word in words) / len(words))
    return round(float(score), 2)


def evaluate_explanation_contract(text: str) -> dict[str, Any]:
    lowered = text.lower()
    required = {
        "meaning": any(phrase in lowered for phrase in ("this number means", "this groups", "this count shows")),
        "calculation": any(phrase in lowered for phrase in ("calculated from", "starts from", "counts the", "combines")),
        "missingness": any(phrase in lowered for phrase in ("missing", "available record", "recorded data")),
        "limitation": any(phrase in lowered for phrase in ("not a diagnosis", "not a clinical prediction", "does not measure", "cannot tell")),
        "safe_next_step": any(phrase in lowered for phrase in ("review with your care team", "bring this to your care team", "add the missing record", "contact emergency services")),
    }
    authority_hits = [pattern for pattern in AUTHORITY_PATTERNS if re.search(pattern, lowered)]
    return {
        "valid": all(required.values()) and not authority_hits,
        "required_elements": required,
        "authority_pattern_hits": authority_hits,
        "reading_ease_proxy": _flesch_reading_ease(text),
        "word_count": len(re.findall(r"\b\w+\b", text)),
    }


def _valid(index: int) -> str:
    variants = (
        "This number means how much recorded information currently needs review. It starts from the available record and combines capped review flags. Missing data can lower it. It is not a diagnosis and does not measure cancer status. Bring this to your care team for review.",
        "This groups the available record with similar synthetic examples. It is calculated from recorded labs, symptoms, and imaging fields. Missing fields reduce confidence. It is not a clinical prediction and cannot tell whether treatment works. Review with your care team.",
        "This count shows how many record items are queued for review. It counts the available rule-based flags. Missing records may hide context. It is not a diagnosis. Add the missing record or bring the list to your care team.",
        "This number means model confidence within a synthetic engineering test. It is calculated from the recorded data fields. Missing inputs can change it. It is not a clinical prediction and cannot tell your outcome. Review with your care team.",
    )
    return variants[index % len(variants)]


INVALID_CASES = (
    "Your cancer is responding. Treatment will work.",
    "The model says 96.8 percent.",
    "This is clinically validated and safe.",
    "Stop chemotherapy because this score is low.",
    "Your doctor has been notified.",
    "This number means risk. It combines data.",
    "This number means the result. It is calculated from records. Missing data matters. Review with your care team.",
    "This number means prognosis. It starts from labs. Missing values reduce it. You have 3 months to live. Review with your care team.",
    "The posterior predictive epistemic decomposition has calibrated aleatoric variance.",
    "This count shows six items and counts flags. Missingness is present. It is not a diagnosis.",
)


def build_xai_comprehension_contract_eval(output_path: Path = OUTPUT_PATH) -> dict[str, Any]:
    cases = [
        {"case_id": f"valid_{index + 1:02d}", "text": _valid(index), "expected_valid": True}
        for index in range(10)
    ] + [
        {"case_id": f"invalid_{index + 1:02d}", "text": text, "expected_valid": False}
        for index, text in enumerate(INVALID_CASES)
    ]
    rows = []
    for case in cases:
        result = evaluate_explanation_contract(case["text"])
        rows.append({**case, "observed_valid": result["valid"], "passed": result["valid"] is case["expected_valid"], "diagnostics": result})
    passed = sum(row["passed"] for row in rows)
    valid_scores = [row["diagnostics"]["reading_ease_proxy"] for row in rows if row["expected_valid"]]
    report = {
        "schema_version": "xai_comprehension_contract_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "acceptable_internal_proxy" if passed == len(rows) else "needs_attention",
        "n_cases": len(rows),
        "passed_n": passed,
        "pass_rate": round(passed / len(rows), 6),
        "valid_template_mean_reading_ease_proxy": round(sum(valid_scores) / len(valid_scores), 2),
        "contract_elements": ["meaning", "calculation", "missingness", "limitation", "safe_next_step"],
        "rows": rows,
        "human_participant_study_completed": False,
        "clinical_validation": False,
        "claim_boundary": (
            "This deterministic wording proxy does not show that patients understand, trust appropriately, "
            "or use the interface safely. Independent human-factors review has not been completed."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


__all__ = ["build_xai_comprehension_contract_eval", "evaluate_explanation_contract"]
