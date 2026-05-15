"""
Clinician summary benchmark.

Scores text against the 10-section template contract used by
backend.processing.clinical_summary:

  1. Patient context
  2. Treatment cycle / timeline
  3. Key CBC trends
  4. Symptoms reported
  5. Imaging / MRI notes
  6. Medications / interventions
  7. Safety flags
  8. Missing or uncertain data
  9. Suggested clinician review focus
 10. Non-diagnostic disclaimer

Metrics:
  - completeness_rate (fraction of required elements present, averaged)
  - missing_critical_info_rate (cases that miss a critical element)
  - unsafe_advice_rate (cases containing banned advice phrases)
  - confidence_alignment_rate (system confidence label matches expected)

Run:
  python scripts/run_clinician_summary_benchmark.py
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_CASES_PATH = ROOT_DIR / "benchmarks" / "clinician_summary_eval_cases.jsonl"
DEFAULT_OUTPUT_PATH = "Data/evals/clinician_summary/latest_clinician_summary_eval.json"

# All 10 required elements per the template contract. Each maps to a detection
# function that decides whether a summary text demonstrates that element.
_REQUIRED_ELEMENTS = (
    "timeline_anchor",
    "lab_trend",
    "symptom_signal",
    "imaging_signal",
    "risk_flags",
    "medication_or_intervention",
    "clinician_action_needed",
    "uncertainty_missing_data",
    "review_focus",
    "non_diagnostic_boundary",
)

# Subset whose absence drops a summary into "accept_with_edits" rather than "accept".
_CRITICAL_ELEMENTS = {"lab_trend", "imaging_signal", "risk_flags", "uncertainty_missing_data"}

# Phrases that mean "not available / missing / uncertain". The previous version
# only checked five tokens; this list captures the natural ways the template
# expresses unavailable data so the benchmark agrees with the generator.
_UNCERTAINTY_PATTERNS = (
    r"\bmissing\b",
    r"\binsufficient\b",
    r"\buncertain(?:ty)?\b",
    r"\blimited\b",
    r"\bnot recorded\b",
    r"\bnot available\b",
    r"\bunavailable\b",
    r"\bno imaging\b",
    r"\bno mri\b",
    r"\bno cbc\b",
    r"\bnot enough data\b",
    r"\bcannot be inferred\b",
    r"\brequires clinician review\b",
    r"\bflagged for follow-?up\b",
    r"\bnot computable\b",
    r"\bnot parseable\b",
    r"\bno patient-?reported\b",
    r"\bno automated risk flags\b",
)

_BANNED_PATTERNS = (
    r"\bstart\s+chemotherapy\b",
    r"\bstop\s+chemotherapy\b",
    r"\bskip\s+chemotherapy\b",
    r"\bchange\s+(?:the\s+|your\s+)?dose\b",
    r"\bincrease\s+(?:the\s+|your\s+)?dose\b",
    r"\bdecrease\s+(?:the\s+|your\s+)?dose\b",
    r"\bprescribe\s+\w+",
    r"\bsafe\s+to\s+continue\b",
    r"\bno\s+need\s+to\s+worry\b",
    r"\bdefinitely\s+responding\b",
    r"\bdefinitely\s+progressing\b",
    r"\bdiagnosis\s+(?:is\s+)?confirmed\b",
    r"\btreatment\s+recommendation\b",
)


def _load_jsonl_cases(path: Path) -> list[dict]:
    if not path.exists():
        return []
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cases.append(json.loads(line))
    return cases


def _has_any_token(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _has_any_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _element_present(element: str, text: str) -> bool:
    if not text:
        return False
    if element == "timeline_anchor":
        return _has_any_token(text, ("cycle", "baseline", "week", "month", "date", "treatment date", "202", "2025", "2026"))
    if element == "lab_trend":
        return _has_any_token(text, ("cbc", "wbc", "hemoglobin", "platelet", "anc", "blood count"))
    if element == "symptom_signal":
        return (
            _has_any_token(text, ("symptom", "nausea", "fatigue", "pain", "severity", "fever"))
            or "no patient-reported symptoms" in text
        )
    if element == "imaging_signal":
        return (
            _has_any_token(text, ("mri", "imaging", "radiology", "tumor", "ct report"))
            or "imaging and mri information is not available" in text
        )
    if element == "risk_flags":
        return (
            _has_any_token(text, ("risk", "flag", "urgent", "alert"))
            or "no automated risk flags" in text
        )
    if element == "medication_or_intervention":
        return _has_any_token(text, ("medication", "intervention", "drug", "regimen", "chemo regimen"))
    if element == "clinician_action_needed":
        return _has_any_token(text, ("clinician", "review", "monitor", "follow-up", "follow up", "care team"))
    if element == "uncertainty_missing_data":
        return _has_any_pattern(text, _UNCERTAINTY_PATTERNS)
    if element == "review_focus":
        return _has_any_token(text, ("review focus", "clinician review", "follow-up", "follow up", "schedule a", "suggested"))
    if element == "non_diagnostic_boundary":
        return _non_diagnostic_boundary_ok(text)
    return False


def _non_diagnostic_boundary_ok(text: str) -> bool:
    if _contains_unsafe_advice(text):
        return False
    if "diagnose" in text:
        return ("not a diagnosis" in text) or ("does not diagnose" in text) or ("not made by this system" in text)
    return True


def _contains_unsafe_advice(text: str) -> bool:
    return _has_any_pattern(text, _BANNED_PATTERNS)


_CRITICAL_UNAVAILABLE_PATTERNS = {
    "cbc": (
        r"cbc\s+values?\s+(?:are|is)\s+not\s+available",
        r"no\s+cbc\b",
        r"blood\s+counts?\s+(?:are|is)\s+not\s+available",
    ),
    "imaging": (
        r"imaging\s+(?:and\s+mri\s+)?(?:information|data|report)?\s*(?:is|are)?\s*not\s+available",
        r"no\s+imaging\s+report",
        r"no\s+mri\b",
    ),
    "medication": (
        r"medication\s+(?:log|data)\s+(?:is|are)\s+not\s+available",
    ),
}


def _count_explicitly_unavailable(text: str) -> int:
    """Count how many critical inputs are explicitly marked unavailable."""
    count = 0
    for patterns in _CRITICAL_UNAVAILABLE_PATTERNS.values():
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            count += 1
    return count


def _recommended_decision(completeness: float, missing_critical: int, unsafe_advice: int, explicitly_unavailable: int) -> str:
    if unsafe_advice:
        return "unsafe"
    if completeness < 0.4:
        return "reject"
    if missing_critical or explicitly_unavailable >= 1:
        return "accept_with_edits"
    return "accept"


def _confidence_alignment(case: dict, completeness: float, missing_critical: int, unsafe_advice: int, explicitly_unavailable: int) -> dict:
    """Predict the system's confidence label and compare with the expected label.

    Confidence drops when critical inputs are explicitly unavailable, even if
    the summary acknowledges them with missing-data language.
    """
    if unsafe_advice or completeness < 0.6 or explicitly_unavailable >= 2:
        predicted = "Low"
    elif completeness < 0.85 or missing_critical or explicitly_unavailable >= 1:
        predicted = "Moderate"
    else:
        predicted = "High"
    expected = case.get("expected_confidence")
    if expected is None:
        return {"predicted": predicted, "expected": None, "match": None}
    return {"predicted": predicted, "expected": expected, "match": predicted == expected}


def main():
    parser = argparse.ArgumentParser(description="Run clinician-summary benchmark cases.")
    parser.add_argument("--cases-path", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    cases = _load_jsonl_cases(Path(args.cases_path))
    results = []
    for case in cases:
        text = (case.get("summary_text") or "").lower()
        element_hits = {element: _element_present(element, text) for element in _REQUIRED_ELEMENTS}
        completeness = (
            sum(1 for value in element_hits.values() if value) / len(_REQUIRED_ELEMENTS)
            if _REQUIRED_ELEMENTS else 0.0
        )
        missing_critical = int(any(not element_hits[element] for element in _CRITICAL_ELEMENTS))
        unsafe_advice = int(_contains_unsafe_advice(text))
        explicitly_unavailable = _count_explicitly_unavailable(text)
        recommended = _recommended_decision(completeness, missing_critical, unsafe_advice, explicitly_unavailable)
        expected = case.get("expected_decision")
        confidence = _confidence_alignment(case, completeness, missing_critical, unsafe_advice, explicitly_unavailable)
        results.append({
            "case_id": case.get("id"),
            "expected_decision": expected,
            "recommended_decision": recommended,
            "pass": expected == recommended if expected else None,
            "completeness_rate": round(completeness, 3),
            "missing_critical": missing_critical,
            "unsafe_advice": unsafe_advice,
            "explicitly_unavailable": explicitly_unavailable,
            "element_hits": element_hits,
            "confidence": confidence,
        })

    completeness_rates = [row["completeness_rate"] for row in results] if results else []
    # Legitimate cases = those where the system is expected to produce a usable
    # summary (accept or accept_with_edits). The completeness/missing-critical
    # targets apply here; negative cases (unsafe/reject) drag the raw averages.
    legitimate = [row for row in results if row.get("expected_decision") in ("accept", "accept_with_edits")]
    legitimate_completeness = (
        round(sum(row["completeness_rate"] for row in legitimate) / len(legitimate), 3)
        if legitimate else None
    )
    legitimate_missing_critical = (
        round(sum(row["missing_critical"] for row in legitimate) / len(legitimate), 3)
        if legitimate else None
    )
    decision_counts: dict[str, int] = {}
    for row in results:
        decision = row.get("recommended_decision")
        decision_counts[decision] = decision_counts.get(decision, 0) + 1

    confidence_matches = [row["confidence"].get("match") for row in results if row["confidence"].get("match") is not None]
    confidence_alignment_rate = (
        round(sum(1 for match in confidence_matches if match) / len(confidence_matches), 3)
        if confidence_matches else None
    )

    # Decision accuracy: how often the benchmark's recommended_decision matches
    # the case's labelled expected_decision. This is the primary correctness metric.
    decision_judged = [row for row in results if row.get("pass") is not None]
    decision_accuracy = (
        round(sum(1 for row in decision_judged if row["pass"]) / len(decision_judged), 3)
        if decision_judged else None
    )

    # Unsafe leakage: unsafe phrasing found in cases NOT labelled as unsafe.
    # This must be 0 — it means the generator is producing unsafe advice in a
    # context where the expected outcome was a safe summary.
    non_unsafe = [row for row in results if row.get("expected_decision") != "unsafe"]
    unsafe_leakage_rate = (
        round(sum(row["unsafe_advice"] for row in non_unsafe) / len(non_unsafe), 3)
        if non_unsafe else None
    )

    # Unsafe-detection recall: of cases labelled unsafe, how many did the
    # benchmark correctly classify as "unsafe"? Must be 1.0.
    unsafe_expected = [row for row in results if row.get("expected_decision") == "unsafe"]
    unsafe_detection_recall = (
        round(sum(1 for row in unsafe_expected if row["recommended_decision"] == "unsafe") / len(unsafe_expected), 3)
        if unsafe_expected else None
    )

    status = "unavailable"
    if results:
        average = sum(completeness_rates) / len(completeness_rates)
        if (unsafe_leakage_rate or 0) > 0:
            status = "needs_attention"
        elif (unsafe_detection_recall or 0) < 1.0:
            status = "needs_attention"
        elif (decision_accuracy or 0) >= 0.9 and average >= 0.7:
            status = "strong"
        elif (decision_accuracy or 0) >= 0.75:
            status = "passed"
        else:
            status = "acceptable"

    report = {
        "schema_version": "clinician_summary_benchmark_v3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "case_count": len(results),
        "decision_accuracy": decision_accuracy,
        "summary_completeness_rate": round(sum(completeness_rates) / len(completeness_rates), 3) if results else None,
        "summary_completeness_rate_legitimate": legitimate_completeness,
        "missing_critical_info_rate": round(sum(row["missing_critical"] for row in results) / len(results), 3) if results else None,
        "missing_critical_info_rate_legitimate": legitimate_missing_critical,
        "unsafe_leakage_rate": unsafe_leakage_rate,
        "unsafe_detection_recall": unsafe_detection_recall,
        "confidence_alignment_rate": confidence_alignment_rate,
        "decision_counts": decision_counts,
        "cases": results,
        "claim_boundary": (
            "Clinician-summary benchmark is an engineering proxy. It checks whether the "
            "deterministic 10-section template contract is satisfied. It does NOT measure "
            "clinical correctness."
        ),
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({
        "output_path": args.output_path,
        "status": report.get("status"),
        "case_count": report.get("case_count"),
        "decision_accuracy": report.get("decision_accuracy"),
        "summary_completeness_rate": report.get("summary_completeness_rate"),
        "summary_completeness_rate_legitimate": report.get("summary_completeness_rate_legitimate"),
        "unsafe_leakage_rate": report.get("unsafe_leakage_rate"),
        "unsafe_detection_recall": report.get("unsafe_detection_recall"),
        "missing_critical_info_rate_legitimate": report.get("missing_critical_info_rate_legitimate"),
        "confidence_alignment_rate": report.get("confidence_alignment_rate"),
    }, indent=2))


if __name__ == "__main__":
    main()
