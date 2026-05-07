import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = ROOT_DIR / "Data" / "agent_eval" / "eval_catalog_coverage.json"


def main():
    eval_dir = ROOT_DIR / "evals"
    rag_catalog = _load_json(eval_dir / "rag_cases.json")
    safety_catalog = _load_json(eval_dir / "safety_cases.json")
    summary_rubric = _load_json(eval_dir / "summary_quality_rubric.json")
    mle_targets = _load_json(eval_dir / "mle_readiness_targets.json")
    agent_report = _load_json(ROOT_DIR / "Data" / "agent_eval" / "latest_agent_regression.json")
    mle_report = _load_json(ROOT_DIR / "Data" / "mle_monitoring" / "latest_mle_readiness.json")

    validation_errors = []
    validation_errors.extend(_validate_case_catalog("rag", rag_catalog))
    validation_errors.extend(_validate_case_catalog("safety", safety_catalog))
    validation_errors.extend(_validate_summary_rubric(summary_rubric))
    validation_errors.extend(_validate_mle_targets(mle_targets))

    regression_cases = {case.get("id"): case for case in agent_report.get("cases") or []}
    expected_automated_cases = _automated_case_ids(rag_catalog) | _automated_case_ids(safety_catalog)
    missing_automated_cases = sorted(case_id for case_id in expected_automated_cases if case_id not in regression_cases)
    failed_automated_cases = sorted(
        case_id
        for case_id in expected_automated_cases
        if case_id in regression_cases and regression_cases[case_id].get("status") != "passed"
    )

    output = {
        "schema_version": "eval_catalog_coverage_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Verifies that documented RAG/safety eval catalogs are wired to automated regression reports.",
        "status": _coverage_status(
            validation_errors=validation_errors,
            missing_automated_cases=missing_automated_cases,
            failed_automated_cases=failed_automated_cases,
            agent_report=agent_report,
            mle_report=mle_report,
        ),
        "catalogs": {
            "rag_cases": {
                "schema_version": rag_catalog.get("schema_version"),
                "case_count": len(rag_catalog.get("cases") or []),
                "automated_case_count": len(_automated_case_ids(rag_catalog)),
            },
            "safety_cases": {
                "schema_version": safety_catalog.get("schema_version"),
                "case_count": len(safety_catalog.get("cases") or []),
                "automated_case_count": len(_automated_case_ids(safety_catalog)),
            },
            "summary_quality_rubric": {
                "schema_version": summary_rubric.get("schema_version"),
                "required_element_count": len(summary_rubric.get("required_summary_elements") or []),
                "metric_count": len(summary_rubric.get("metrics") or []),
                "automation_status": "rubric_only_clinician_feedback_proxy",
            },
            "mle_readiness_targets": {
                "schema_version": mle_targets.get("schema_version"),
                "strict_status": mle_report.get("status"),
                "poc_demo_readiness": (mle_report.get("poc_demo_readiness") or {}).get("status"),
                "hard_gate_status": mle_report.get("hard_gate_status"),
            },
        },
        "automated_regression": {
            "report_status": (agent_report.get("summary") or {}).get("status"),
            "case_count": agent_report.get("case_count"),
            "expected_catalog_case_count": len(expected_automated_cases),
            "missing_automated_cases": missing_automated_cases,
            "failed_automated_cases": failed_automated_cases,
        },
        "validation_errors": validation_errors,
        "claim_boundary": "Eval catalog coverage is engineering evidence only, not clinical validation.",
    }

    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps({
        "output_path": str(DEFAULT_OUTPUT_PATH.relative_to(ROOT_DIR)),
        "status": output["status"],
        "rag_cases": output["catalogs"]["rag_cases"]["case_count"],
        "safety_cases": output["catalogs"]["safety_cases"]["case_count"],
        "missing_automated_cases": missing_automated_cases,
        "failed_automated_cases": failed_automated_cases,
    }, indent=2))

    if output["status"] == "failed":
        raise SystemExit(1)


def _load_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_case_catalog(name, catalog):
    errors = []
    if not catalog.get("schema_version"):
        errors.append(f"{name}: missing schema_version")
    cases = catalog.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append(f"{name}: missing non-empty cases list")
        return errors
    seen = set()
    for index, case in enumerate(cases, start=1):
        case_id = case.get("id")
        if not case_id:
            errors.append(f"{name}: case {index} missing id")
        elif case_id in seen:
            errors.append(f"{name}: duplicate case id {case_id}")
        seen.add(case_id)
        if not case.get("input"):
            errors.append(f"{name}: case {case_id or index} missing input")
    return errors


def _validate_summary_rubric(rubric):
    errors = []
    if not rubric.get("schema_version"):
        errors.append("summary_quality_rubric: missing schema_version")
    if not rubric.get("required_summary_elements"):
        errors.append("summary_quality_rubric: missing required_summary_elements")
    if not rubric.get("metrics"):
        errors.append("summary_quality_rubric: missing metrics")
    return errors


def _validate_mle_targets(targets):
    errors = []
    if not targets.get("schema_version"):
        errors.append("mle_readiness_targets: missing schema_version")
    if not targets.get("required_for_poc_demo"):
        errors.append("mle_readiness_targets: missing required_for_poc_demo")
    return errors


def _automated_case_ids(catalog):
    ids = set()
    for case in catalog.get("cases") or []:
        ids.add(case.get("automated_case_id") or case.get("id"))
    return {case_id for case_id in ids if case_id}


def _coverage_status(validation_errors, missing_automated_cases, failed_automated_cases, agent_report, mle_report):
    if validation_errors or failed_automated_cases:
        return "failed"
    if missing_automated_cases:
        return "unideal"
    if (agent_report.get("summary") or {}).get("status") == "strong" and mle_report.get("hard_gate_status") == "passed":
        return "strong"
    return "acceptable"


if __name__ == "__main__":
    main()
