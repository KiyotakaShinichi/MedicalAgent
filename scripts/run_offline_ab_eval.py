"""Offline A/B evaluation runner.

Loads ``config/ab_tests.yaml``, resolves each test's baseline + candidate
to a callable, runs the cases through both, scores them with
``backend.services.ab_testing.run_ab_test``, and writes one aggregated
artifact to ``Data/evals/ab_tests/latest_ab_test_report.json``.

Variant resolution
~~~~~~~~~~~~~~~~~~
The runner ships with three deterministic builtin variants so the
framework can be exercised end-to-end without spinning up the live
agent stack:

  - ``builtin_safety_baseline``  — emits a refusal envelope for cases
    that ``expects_refusal``, an educational envelope otherwise.  This
    is the reference SAFE variant.
  - ``builtin_unsafe_demo``      — emits unsafe treatment-directive
    wording on refusal cases.  Used by the regression test that the
    framework must REJECT it.
  - ``builtin_silent_demo``      — emits empty replies.  Used by the
    test that a candidate which simply omits the escalation phrase
    must be HOLD or REJECT.

Live-agent variants
~~~~~~~~~~~~~~~~~~~
Adding a real-agent variant would import
``backend.services.agent_rag.run_patient_agent_pipeline`` and wrap it
into a ``(case) -> envelope`` callable.  Not done by default because
the live agent requires a SQLAlchemy session and is slow; the
framework is deliberately decoupled so callers can plug it in.

Usage
~~~~~
    python scripts/run_offline_ab_eval.py
    python scripts/run_offline_ab_eval.py --config config/ab_tests.yaml
    python scripts/run_offline_ab_eval.py --json

Exit codes
~~~~~~~~~~
  0  every test matches its expected decision contract
  1  at least one unexpected A/B decision or safety regression
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.ab_testing import (  # noqa: E402
    AbTestConfig,
    AbTestReport,
    run_ab_test,
    write_report,
)

DEFAULT_CONFIG = ROOT / "config" / "ab_tests.yaml"
DEFAULT_OUTPUT = ROOT / "Data" / "evals" / "ab_tests" / "latest_ab_test_report.json"


# ─── Builtin deterministic variants ──────────────────────────────────────────


_SAFE_EDU_REPLY = (
    "General information: this is patient-education context. "
    "Discuss personal decisions with your oncology care team."
)
_SAFE_REFUSAL_REPLY = (
    "I cannot diagnose, recommend treatment, or change your medication. "
    "Please contact your oncology care team — they can review this in "
    "your full clinical context. If symptoms feel sudden, severe, or "
    "unsafe, use your local emergency services."
)
_MISSING_DATA_REPLY = (
    "I don't have enough report detail to summarize this. Please paste "
    "the findings or impression text and I can organize it for clinician "
    "review."
)


def builtin_safety_baseline(case: Mapping[str, Any]) -> dict[str, Any]:
    if case.get("expects_refusal"):
        return {"reply": _SAFE_REFUSAL_REPLY, "citations": [], "abstained": False}
    if case.get("expects_missing_data_disclosure"):
        return {"reply": _MISSING_DATA_REPLY, "citations": [], "abstained": True}
    if case.get("expects_citation"):
        return {
            "reply": _SAFE_EDU_REPLY,
            "citations": [{"id": "cdc-fever-chemo", "title": "CDC fever guidance"}],
            "abstained": False,
        }
    return {"reply": _SAFE_EDU_REPLY, "citations": [], "abstained": False}


def builtin_unsafe_demo(case: Mapping[str, Any]) -> dict[str, Any]:
    """Deliberately unsafe — emits blocked treatment-directive wording
    on refusal cases.  Framework must REJECT this candidate."""
    if case.get("expects_refusal"):
        return {
            "reply": "You should stop chemotherapy because your counts are low.",
            "citations": [],
            "abstained": False,
        }
    return builtin_safety_baseline(case)


def builtin_silent_demo(case: Mapping[str, Any]) -> dict[str, Any]:
    """Returns empty replies — should fail refusal_correctness without
    being unsafe-leaky.  Decision: HOLD or REJECT (refusal regression)."""
    return {"reply": "", "citations": [], "abstained": False}


VARIANTS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "builtin_safety_baseline": builtin_safety_baseline,
    "builtin_unsafe_demo":     builtin_unsafe_demo,
    "builtin_silent_demo":     builtin_silent_demo,
}


# ─── Runner ──────────────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for the A/B test config.") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"A/B test config must be an object: {path}")
    return payload


def _resolve_variant(name: str) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    if name not in VARIANTS:
        raise KeyError(f"Unknown variant {name!r}; known: {sorted(VARIANTS)}")
    return VARIANTS[name]


def run_all_tests(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_yaml(config_path)
    case_sets = config.get("case_sets") or {}
    tests = config.get("tests") or []
    reports: list[dict[str, Any]] = []
    expectation_rows: list[dict[str, Any]] = []

    for test_def in tests:
        case_set_name = test_def["case_set"]
        case_set = case_sets.get(case_set_name)
        if not case_set:
            raise KeyError(f"Test {test_def['name']!r} references unknown case_set {case_set_name!r}")
        baseline = _resolve_variant(test_def["baseline"])
        candidate = _resolve_variant(test_def["candidate"])
        report = run_ab_test(
            cases=list(case_set["cases"]),
            baseline=baseline,
            candidate=candidate,
            config=AbTestConfig(
                name=test_def["name"],
                description=test_def.get("description", ""),
            ),
        )
        expected = _expected_decisions(test_def)
        expectation_passed = report.decision in expected if expected else report.decision != "REJECT"
        report_dict = report.to_dict()
        report_dict["expected_decisions"] = sorted(expected) if expected else ["PROMOTE", "HOLD"]
        report_dict["negative_control"] = bool(test_def.get("negative_control", False))
        report_dict["expectation_passed"] = expectation_passed
        reports.append(report_dict)
        expectation_rows.append({
            "name": report.name,
            "decision": report.decision,
            "expected_decisions": sorted(expected) if expected else ["PROMOTE", "HOLD"],
            "negative_control": bool(test_def.get("negative_control", False)),
            "expectation_passed": expectation_passed,
        })

    overall_decision = _overall_decision([r["decision"] for r in reports])
    unexpected = [row for row in expectation_rows if not row["expectation_passed"]]
    status = "strong" if not unexpected else "failed"
    return {
        "schema_version": "ab_test_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.relative_to(ROOT)),
        "status": status,
        "test_count": len(reports),
        "overall_decision": overall_decision,
        "expectations": {
            "passed": len(expectation_rows) - len(unexpected),
            "failed": len(unexpected),
            "rows": expectation_rows,
        },
        "reports": reports,
        "claim_boundary": config.get("claim_boundary", ""),
    }


def _expected_decisions(test_def: Mapping[str, Any]) -> set[str]:
    if "expected_decision" in test_def:
        return {str(test_def["expected_decision"])}
    if "expected_decisions" in test_def:
        return {str(item) for item in test_def["expected_decisions"]}
    return set()


def _overall_decision(decisions: list[str]) -> str:
    if any(d == "REJECT" for d in decisions):
        return "REJECT"
    if any(d == "HOLD" for d in decisions):
        return "HOLD"
    return "PROMOTE"


def _print_human(report: dict[str, Any]) -> None:
    print(f"OncoTrack offline A/B suite: {report['status'].upper()} ({report['overall_decision']})")
    print(f"  config: {report['config_path']}")
    print(f"  tests:  {report['test_count']}")
    print(f"  expectations: {report['expectations']['passed']} passed / {report['expectations']['failed']} failed")
    print()
    for r in report["reports"]:
        b = r["baseline"]
        c = r["candidate"]
        expectation = "ok" if r.get("expectation_passed") else "unexpected"
        print(f"  [{r['decision']:7}] {r['name']} ({expectation})")
        print(f"          unsafe_leak: {b['unsafe_leakage_rate']:.3f} ->{c['unsafe_leakage_rate']:.3f}")
        print(f"          refusal:     {b['refusal_correctness']:.3f} ->{c['refusal_correctness']:.3f}")
        print(f"          claim_bound: {b['claim_boundary_compliance']:.3f} ->{c['claim_boundary_compliance']:.3f}")
        for reason in r["reasons"]:
            print(f"          - {reason}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OncoTrack offline A/B evaluation runner")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    report = run_all_tests(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_human(report)
        print(f"\n  wrote {args.output.relative_to(ROOT)}")
    return 0 if report["status"] == "strong" else 1


if __name__ == "__main__":
    raise SystemExit(main())
