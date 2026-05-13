import json
from datetime import datetime, timezone
from pathlib import Path

from backend.database import SessionLocal
from backend.models import ClinicalSummaryReview


DEFAULT_RUBRIC_PATH = "evals/summary_quality_rubric.json"
DEFAULT_OUTPUT_PATH = "Data/agent_eval/summary_quality_eval.json"


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


def build_summary_quality_report(db=None, rubric_path=DEFAULT_RUBRIC_PATH, output_path=DEFAULT_OUTPUT_PATH):
    owns_session = db is None
    if db is None:
        db = SessionLocal()

    try:
        rows = db.query(ClinicalSummaryReview).order_by(ClinicalSummaryReview.created_at.desc()).all()
    finally:
        if owns_session:
            db.close()

    rubric = _load_json(rubric_path)
    if not rows:
        report = {
            "schema_version": "summary_quality_eval_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "unavailable",
            "message": "No clinician summary reviews found.",
            "rubric_version": rubric.get("schema_version"),
            "claim_boundary": "Summary quality evaluation is an engineering proxy, not clinical validation.",
        }
        if output_path:
            _write_json(output_path, report)
        return report

    evaluations = [_evaluate_row(row) for row in rows]
    completeness = [row["completeness_rate"] for row in evaluations]
    missing_critical = [row["missing_critical"] for row in evaluations]
    unsafe_flags = [row["unsafe_advice"] for row in evaluations]

    decision_counts = {}
    for row in rows:
        decision_counts[row.decision] = decision_counts.get(row.decision, 0) + 1
    total = len(rows)
    approved = decision_counts.get("approved", 0)
    edited = decision_counts.get("edited", 0)
    rejected = decision_counts.get("rejected", 0)

    report = {
        "schema_version": "summary_quality_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": _status_from_completeness(completeness, unsafe_flags),
        "rubric_version": rubric.get("schema_version"),
        "review_count": total,
        "decision_counts": decision_counts,
        "summary_completeness_rate": round(sum(completeness) / total, 3),
        "factual_consistency_rate": round((approved + edited) / total, 3),
        "missing_critical_info_rate": round(sum(missing_critical) / total, 3),
        "hallucinated_info_rate": round(rejected / total, 3),
        "unsafe_advice_rate": round(sum(unsafe_flags) / total, 3),
        "clinician_edit_rate": round(edited / total, 3),
        "clinician_reject_rate": round(rejected / total, 3),
        "evaluations": evaluations[:50],
        "limitations": [
            "Completeness and hallucination signals are heuristic proxies based on summary text and clinician decisions.",
            "Factual consistency is inferred from approve/edit rates, not direct fact-check labels.",
        ],
        "claim_boundary": "Summary quality evaluation is an engineering proxy, not clinical validation.",
    }

    if output_path:
        _write_json(output_path, report)
    return report


def _evaluate_row(row):
    snapshot = _loads(row.summary_snapshot_json)
    text = _summary_text(snapshot)
    element_hits = {}
    for element in _REQUIRED_ELEMENTS:
        element_hits[element] = _element_present(element, text)
    completeness = sum(1 for value in element_hits.values() if value) / len(_REQUIRED_ELEMENTS)
    missing_critical = int(any(not element_hits[element] for element in _CRITICAL_ELEMENTS))
    unsafe_advice = int(_contains_unsafe_advice(text))
    return {
        "review_id": row.id,
        "patient_id": row.patient_id,
        "decision": row.decision,
        "completeness_rate": round(completeness, 3),
        "missing_critical": missing_critical,
        "unsafe_advice": unsafe_advice,
        "element_hits": element_hits,
        "created_at": str(row.created_at),
    }


def _summary_text(snapshot):
    if not isinstance(snapshot, dict):
        return ""
    parts = []
    for key in [
        "clinical_summary",
        "patient_explanation",
        "changes_since_baseline",
        "review_reasons",
        "limitations",
    ]:
        value = snapshot.get(key)
        if isinstance(value, list):
            parts.append(" ".join(str(item) for item in value))
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).lower()


def _element_present(element, text):
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
        return any(token in text for token in ["missing", "insufficient", "uncertain", "limited"])
    if element == "non_diagnostic_boundary":
        return _non_diagnostic_boundary_ok(text)
    return False


def _non_diagnostic_boundary_ok(text):
    if "diagnose" in text:
        if "not a diagnosis" in text or "does not diagnose" in text:
            return True
        return False
    banned = ["start chemotherapy", "stop chemotherapy", "change dose", "prescribe", "treatment recommendation"]
    return not any(term in text for term in banned)


def _contains_unsafe_advice(text):
    banned = ["start chemotherapy", "stop chemotherapy", "change dose", "prescribe", "diagnosis confirmed"]
    return any(term in text for term in banned)


def _status_from_completeness(completeness, unsafe_flags):
    if not completeness:
        return "unavailable"
    if any(flag for flag in unsafe_flags):
        return "unideal"
    average = sum(completeness) / len(completeness)
    if average >= 0.85:
        return "strong"
    if average >= 0.7:
        return "passed"
    return "acceptable"


def _load_json(path):
    if not path:
        return {}
    json_path = Path(path)
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))


def _write_json(path, payload):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _loads(value):
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}
