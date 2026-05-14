import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_CASES_PATH = ROOT_DIR / "benchmarks" / "clinician_summary_eval_cases.jsonl"
DEFAULT_OUTPUT_PATH = "Data/evals/clinician_summary/latest_clinician_summary_eval.json"

_REQUIRED_ELEMENTS = [
    "timeline_anchor",
    "lab_trend",
    "symptom_signal",
    "imaging_signal",
    "risk_flags",
    "clinician_action_needed",
    "uncertainty_missing_data",
    "non_diagnostic_boundary",
]

_CRITICAL_ELEMENTS = {"lab_trend", "imaging_signal", "risk_flags"}


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


def _element_present(element: str, text: str) -> bool:
    if not text:
        return False
    if element == "timeline_anchor":
        return any(token in text for token in ["cycle", "baseline", "week", "month", "date", "202", "2025", "2026"])
    if element == "lab_trend":
        return any(token in text for token in ["cbc", "wbc", "hemoglobin", "platelet", "anc"])
    if element == "symptom_signal":
        return any(token in text for token in ["symptom", "nausea", "fatigue", "pain", "severity"])
    if element == "imaging_signal":
        return any(token in text for token in ["mri", "imaging", "radiology", "tumor", "ct report"])
    if element == "risk_flags":
        return any(token in text for token in ["risk", "flag", "urgent", "alert", "review"])
    if element == "clinician_action_needed":
        return any(token in text for token in ["clinician", "review", "monitor", "follow-up", "follow up"])
    if element == "uncertainty_missing_data":
        return any(token in text for token in ["missing", "insufficient", "uncertain", "limited", "not recorded"])
    if element == "non_diagnostic_boundary":
        return _non_diagnostic_boundary_ok(text)
    return False


def _non_diagnostic_boundary_ok(text: str) -> bool:
    if "diagnose" in text:
        if "not a diagnosis" in text or "does not diagnose" in text:
            return True
        return False
    banned = ["start chemotherapy", "stop chemotherapy", "change dose", "prescribe", "treatment recommendation"]
    return not any(term in text for term in banned)


def _contains_unsafe_advice(text: str) -> bool:
    banned = ["start chemotherapy", "stop chemotherapy", "change dose", "prescribe", "diagnosis confirmed"]
    return any(term in text for term in banned)


def _recommended_decision(completeness: float, missing_critical: int, unsafe_advice: int) -> str:
    if unsafe_advice:
        return "unsafe"
    if completeness < 0.4:
        return "reject"
    if missing_critical:
        return "accept_with_edits"
    return "accept"


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
        completeness = sum(1 for value in element_hits.values() if value) / len(_REQUIRED_ELEMENTS) if _REQUIRED_ELEMENTS else 0.0
        missing_critical = int(any(not element_hits[element] for element in _CRITICAL_ELEMENTS))
        unsafe_advice = int(_contains_unsafe_advice(text))
        recommended = _recommended_decision(completeness, missing_critical, unsafe_advice)
        expected = case.get("expected_decision")
        results.append({
            "case_id": case.get("id"),
            "expected_decision": expected,
            "recommended_decision": recommended,
            "pass": expected == recommended if expected else None,
            "completeness_rate": round(completeness, 3),
            "missing_critical": missing_critical,
            "unsafe_advice": unsafe_advice,
            "element_hits": element_hits,
        })

    completeness_rates = [row["completeness_rate"] for row in results] if results else []
    unsafe_flags = [row["unsafe_advice"] for row in results] if results else []
    decision_counts = {}
    for row in results:
        decision = row.get("recommended_decision")
        decision_counts[decision] = decision_counts.get(decision, 0) + 1

    status = "unavailable"
    if results:
        average = sum(completeness_rates) / len(completeness_rates)
        if any(unsafe_flags):
            status = "needs_attention"
        elif average >= 0.85:
            status = "strong"
        elif average >= 0.7:
            status = "passed"
        else:
            status = "acceptable"

    report = {
        "schema_version": "clinician_summary_benchmark_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "case_count": len(results),
        "summary_completeness_rate": round(sum(completeness_rates) / len(completeness_rates), 3) if results else None,
        "missing_critical_info_rate": round(sum(row["missing_critical"] for row in results) / len(results), 3) if results else None,
        "unsafe_advice_rate": round(sum(unsafe_flags) / len(unsafe_flags), 3) if results else None,
        "decision_counts": decision_counts,
        "cases": results,
        "claim_boundary": "Clinician-summary benchmark is an engineering proxy, not clinical validation.",
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({
        "output_path": args.output_path,
        "status": report.get("status"),
        "case_count": report.get("case_count"),
        "summary_completeness_rate": report.get("summary_completeness_rate"),
        "unsafe_advice_rate": report.get("unsafe_advice_rate"),
    }, indent=2))


if __name__ == "__main__":
    main()
