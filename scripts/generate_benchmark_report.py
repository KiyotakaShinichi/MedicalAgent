import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

BENCHMARK_REPORT_PATH = ROOT_DIR / "benchmarks" / "benchmark_report.md"
BENCHMARK_CSV_PATH = ROOT_DIR / "benchmarks" / "benchmark_results.csv"
BENCHMARK_JSON_PATH = ROOT_DIR / "Data" / "evals" / "benchmark" / "latest_benchmark_summary.json"

DEFAULT_PATHS = {
    "safety": "Data/evals/safety/latest_safety_benchmark.json",
    "adversarial": "Data/evals/safety/latest_adversarial_eval.json",
    "rag": "Data/evals/rag/latest_rag_benchmark.json",
    "multilingual": "Data/evals/safety/latest_multilingual_refusal_eval.json",
    "model": "Data/evals/models/latest_model_benchmark.json",
    "realism": "Data/evals/realism/latest_realism_checks.json",
    "realism_report": "Data/mle_monitoring/synthetic_realism_report.json",
    "clinician": "Data/evals/clinician_summary/latest_clinician_summary_eval.json",
    "summary_quality": "Data/agent_eval/summary_quality_eval.json",
}

FALLBACKS = {
    "safety": "Data/evals/safety/latest_safety_red_team.json",
    "rag": "Data/evals/rag/latest_rag_eval.json",
}


def _load_json(path: str) -> dict:
    json_path = ROOT_DIR / path
    if not json_path.exists():
        return {"status": "not_generated", "path": path}
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "error", "path": path}


def _load_with_fallback(primary: str, fallback: str | None = None) -> dict:
    payload = _load_json(primary)
    if payload.get("status") == "not_generated" and fallback:
        return _load_json(fallback)
    return payload


def _safe_rate(value) -> float | None:
    try:
        if value is None:
            return None
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def _category_rate(cases: list[dict], categories: set[str]) -> float | None:
    if not cases:
        return None
    subset = [case for case in cases if case.get("category") in categories]
    if not subset:
        return None
    passed = sum(1 for case in subset if case.get("pass"))
    return round(passed / len(subset), 3)


def _append_row(rows: list[dict], benchmark: str, metric: str, value, status: str, source: str, notes: str = ""):
    rows.append({
        "benchmark": benchmark,
        "metric": metric,
        "value": value,
        "status": status,
        "source_path": source,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
    })


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["benchmark", "metric", "value", "status", "source_path", "generated_at", "notes"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_value(value) -> str:
    if value is None:
        return "not_generated"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main():
    safety = _load_with_fallback(DEFAULT_PATHS["safety"], FALLBACKS.get("safety"))
    adversarial = _load_json(DEFAULT_PATHS["adversarial"])
    rag = _load_with_fallback(DEFAULT_PATHS["rag"], FALLBACKS.get("rag"))
    multilingual = _load_json(DEFAULT_PATHS["multilingual"])
    model = _load_json(DEFAULT_PATHS["model"])
    realism = _load_json(DEFAULT_PATHS["realism"])
    realism_report = _load_json(DEFAULT_PATHS["realism_report"])
    clinician = _load_json(DEFAULT_PATHS["clinician"])
    if clinician.get("status") == "not_generated":
        clinician = _load_json(DEFAULT_PATHS["summary_quality"])

    safety_cases = safety.get("cases") or []
    safety_summary = safety.get("summary") or {}

    urgent_rate = _category_rate(safety_cases, {"urgent_symptom", "urgent_symptom_report", "low_cbc_infection_risk"})
    privacy_rate = _category_rate(safety_cases, {"privacy", "cross_patient_privacy"})
    injection_rate = _category_rate(safety_cases, {"prompt_injection", "encoded_attack", "multilingual_attack"})
    pass_rate = _safe_rate(safety_summary.get("pass_rate"))
    unsafe_pass_rate = round(1 - pass_rate, 3) if pass_rate is not None else None

    rag_summary = rag.get("summary") or {}
    rag_pass_rate = _safe_rate(rag_summary.get("pass_rate"))

    model_status = model.get("status")
    model_champ = None
    for entry in (model.get("synthetic_classification") or []):
        if entry.get("label") == "synthetic_champion":
            model_champ = entry
            break

    realism_status = realism.get("status")
    realism_alignment = _safe_rate(realism_report.get("realism_alignment_score"))

    clinician_status = clinician.get("status") or clinician.get("summary", {}).get("status")
    clinician_completeness = clinician.get("summary_completeness_rate") or clinician.get("summary", {}).get("summary_completeness_rate")
    clinician_unsafe = clinician.get("unsafe_advice_rate") or clinician.get("summary", {}).get("unsafe_advice_rate")

    rows = []
    _append_row(rows, "safety", "unsafe_pass_rate", unsafe_pass_rate, safety_summary.get("status") or safety.get("status"), DEFAULT_PATHS["safety"])
    _append_row(rows, "safety", "urgent_escalation_recall", urgent_rate, safety_summary.get("status") or safety.get("status"), DEFAULT_PATHS["safety"])
    _append_row(rows, "safety", "privacy_leak_rate", None if privacy_rate is None else round(1 - privacy_rate, 3), safety_summary.get("status") or safety.get("status"), DEFAULT_PATHS["safety"])
    _append_row(rows, "safety", "prompt_injection_resistance", injection_rate, safety_summary.get("status") or safety.get("status"), DEFAULT_PATHS["safety"])

    _append_row(rows, "adversarial", "attack_block_rate", _safe_rate((adversarial.get("summary") or {}).get("pass_rate")), adversarial.get("summary", {}).get("status") or adversarial.get("status"), DEFAULT_PATHS["adversarial"])

    _append_row(rows, "rag", "pass_rate", rag_pass_rate, rag_summary.get("status") or rag.get("status"), DEFAULT_PATHS["rag"])
    _append_row(rows, "rag", "citation_coverage", _safe_rate(rag_summary.get("citation_coverage_rate")), rag_summary.get("status") or rag.get("status"), DEFAULT_PATHS["rag"])
    _append_row(rows, "rag", "expected_source_hit", _safe_rate(rag_summary.get("expected_source_hit_rate")), rag_summary.get("status") or rag.get("status"), DEFAULT_PATHS["rag"])
    _append_row(rows, "rag", "refusal_correct", _safe_rate(rag_summary.get("refusal_correct_rate")), rag_summary.get("status") or rag.get("status"), DEFAULT_PATHS["rag"])
    _append_row(rows, "rag", "unsafe_answer_rate", _safe_rate(rag_summary.get("unsafe_answer_rate")), rag_summary.get("status") or rag.get("status"), DEFAULT_PATHS["rag"])

    if model_champ:
        _append_row(rows, "model", "auroc", model_champ.get("roc_auc"), model_status, DEFAULT_PATHS["model"], "synthetic_champion")
        _append_row(rows, "model", "auprc", model_champ.get("auprc"), model_status, DEFAULT_PATHS["model"], "synthetic_champion")
        _append_row(rows, "model", "brier", model_champ.get("brier"), model_status, DEFAULT_PATHS["model"], "synthetic_champion")
        _append_row(rows, "model", "ece_after", model_champ.get("ece_after"), model_status, DEFAULT_PATHS["model"], "synthetic_champion")

    _append_row(rows, "realism", "alignment_score", realism_alignment, realism_status, DEFAULT_PATHS["realism_report"], "synthetic_realism_report")
    _append_row(rows, "realism", "realism_checks_status", realism_status, realism_status, DEFAULT_PATHS["realism"], "realism_checks")

    _append_row(rows, "clinician_summary", "summary_completeness_rate", clinician_completeness, clinician_status, DEFAULT_PATHS["clinician"], "clinician_summary")
    _append_row(rows, "clinician_summary", "unsafe_advice_rate", clinician_unsafe, clinician_status, DEFAULT_PATHS["clinician"], "clinician_summary")

    _write_csv(BENCHMARK_CSV_PATH, rows)

    benchmark_summary = {
        "schema_version": "benchmark_ladder_summary_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "available" if rows else "not_generated",
        "benchmarks": {
            "safety": {
                "status": safety_summary.get("status") or safety.get("status"),
                "unsafe_pass_rate": unsafe_pass_rate,
                "urgent_escalation_recall": urgent_rate,
                "privacy_leak_rate": None if privacy_rate is None else round(1 - privacy_rate, 3),
                "prompt_injection_resistance": injection_rate,
            },
            "adversarial": {
                "status": (adversarial.get("summary") or {}).get("status") or adversarial.get("status"),
                "attack_block_rate": _safe_rate((adversarial.get("summary") or {}).get("pass_rate")),
            },
            "rag": {
                "status": rag_summary.get("status") or rag.get("status"),
                "pass_rate": rag_pass_rate,
                "citation_coverage": _safe_rate(rag_summary.get("citation_coverage_rate")),
                "expected_source_hit": _safe_rate(rag_summary.get("expected_source_hit_rate")),
                "refusal_correct": _safe_rate(rag_summary.get("refusal_correct_rate")),
                "unsafe_answer_rate": _safe_rate(rag_summary.get("unsafe_answer_rate")),
            },
            "model": {
                "status": model_status,
                "synthetic_champion_auroc": (model_champ or {}).get("roc_auc"),
                "synthetic_champion_auprc": (model_champ or {}).get("auprc"),
                "synthetic_champion_brier": (model_champ or {}).get("brier"),
                "synthetic_champion_ece_after": (model_champ or {}).get("ece_after"),
            },
            "realism": {
                "status": realism_status,
                "alignment_score": realism_alignment,
                "realism_checks_status": realism_status,
            },
            "clinician_summary": {
                "status": clinician_status,
                "summary_completeness_rate": clinician_completeness,
                "unsafe_advice_rate": clinician_unsafe,
            },
        },
        "report_path": BENCHMARK_REPORT_PATH.relative_to(ROOT_DIR).as_posix(),
        "csv_path": BENCHMARK_CSV_PATH.relative_to(ROOT_DIR).as_posix(),
        "claim_boundary": (
            "Benchmarks are engineering evidence only; they do not establish clinical validation."
        ),
    }

    BENCHMARK_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_JSON_PATH.write_text(json.dumps(benchmark_summary, indent=2), encoding="utf-8")

    report_lines = [
        "# MedicalAgent Benchmark Report",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Claim boundary: benchmarks are engineering evidence only. They do not establish clinical validation.",
        "",
        "Benchmark philosophy:",
        "MedicalAgent is evaluated not by how often it answers, but by whether it answers only when safe, cites only when grounded, escalates when needed, and exposes uncertainty when the data is weak.",
        "",
        "## Safety benchmark",
        f"- unsafe_pass_rate: {_format_value(unsafe_pass_rate)}",
        f"- urgent_escalation_recall: {_format_value(urgent_rate)}",
        f"- privacy_leak_rate: {_format_value(None if privacy_rate is None else round(1 - privacy_rate, 3))}",
        f"- prompt_injection_resistance: {_format_value(injection_rate)}",
        "",
        "## Adversarial benchmark",
        f"- attack_block_rate: {_format_value((adversarial.get('summary') or {}).get('pass_rate'))}",
        "",
        "## RAG benchmark",
        f"- pass_rate: {_format_value(rag_pass_rate)}",
        f"- citation_coverage: {_format_value(rag_summary.get('citation_coverage_rate'))}",
        f"- expected_source_hit: {_format_value(rag_summary.get('expected_source_hit_rate'))}",
        f"- refusal_correct: {_format_value(rag_summary.get('refusal_correct_rate'))}",
        f"- unsafe_answer_rate: {_format_value(rag_summary.get('unsafe_answer_rate'))}",
        "",
        "## Model benchmark",
        f"- synthetic_champion_auroc: {_format_value((model_champ or {}).get('roc_auc'))}",
        f"- synthetic_champion_auprc: {_format_value((model_champ or {}).get('auprc'))}",
        f"- synthetic_champion_brier: {_format_value((model_champ or {}).get('brier'))}",
        f"- synthetic_champion_ece_after: {_format_value((model_champ or {}).get('ece_after'))}",
        "",
        "## Synthetic realism",
        f"- realism_alignment_score: {_format_value(realism_alignment)}",
        f"- realism_checks_status: {_format_value(realism_status)}",
        "",
        "## Clinician summary benchmark",
        f"- summary_completeness_rate: {_format_value(clinician_completeness)}",
        f"- unsafe_advice_rate: {_format_value(clinician_unsafe)}",
    ]

    BENCHMARK_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "benchmark_report": BENCHMARK_REPORT_PATH.relative_to(ROOT_DIR).as_posix(),
        "benchmark_csv": BENCHMARK_CSV_PATH.relative_to(ROOT_DIR).as_posix(),
        "benchmark_json": BENCHMARK_JSON_PATH.relative_to(ROOT_DIR).as_posix(),
        "row_count": len(rows),
    }, indent=2))


if __name__ == "__main__":
    main()
