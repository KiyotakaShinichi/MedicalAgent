"""MLE promotion decision gate (synthetic-only, read-only).

Aggregates the existing engineering audits under ``Data/evals/models/``
and returns one of three decisions:

  - PROMOTE   — every condition passed
  - HOLD      — non-critical failures only, or critical with ``decision_on_fail: "HOLD"``
  - REJECT    — at least one critical condition with ``decision_on_fail: "REJECT"`` failed

The gate is intentionally **read-only**: it does not retrain, doesn't
mutate any artifact, and doesn't bypass any safety layer.  It exists so
a release reviewer (or future CI step) has a single PROMOTE/HOLD/REJECT
verdict instead of staring at 10 separate audit JSONs.

Claim boundary
~~~~~~~~~~~~~~
This is an engineering promotion gate over synthetic-only audits.
PROMOTE means the engineering proxies are satisfied — it is NOT
clinical validation, regulatory approval, or evidence of real-world
patient benefit.  Production behavior must still pass the deterministic
safety gates, source-governed RAG, the post-generation validator, the
medical claim boundary checker, and clinician review.

Usage
~~~~~
    python scripts/run_mle_promotion_gate.py
    python scripts/run_mle_promotion_gate.py --config config/mle_promotion_thresholds.yaml
    python scripts/run_mle_promotion_gate.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "mle_promotion_thresholds.yaml"
DEFAULT_OUTPUT = ROOT / "Data" / "evals" / "models" / "latest_mle_promotion_gate.json"


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for the MLE promotion gate config.") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"MLE promotion config must be an object: {path}")
    return payload


def _dig(payload: Any, path: list[Any]) -> Any:
    value = payload
    for key in path:
        if isinstance(value, dict):
            value = value.get(key)
        elif isinstance(value, list) and isinstance(key, int) and 0 <= key < len(value):
            value = value[key]
        else:
            return None
    return value


def _compare(actual: Any, op: str | None, expected: Any) -> bool:
    if op in {None, ""}:
        return True
    if actual is None:
        return False
    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    try:
        left = float(actual)
        right = float(expected)
    except (TypeError, ValueError):
        return False
    if op == ">=":
        return left >= right
    if op == ">":
        return left > right
    if op == "<=":
        return left <= right
    if op == "<":
        return left < right
    raise ValueError(f"Unsupported comparator: {op!r}")


def _parse_timestamp(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _check_condition(condition: dict[str, Any]) -> dict[str, Any]:
    artifact_rel = condition["artifact"]
    artifact_path = ROOT / artifact_rel
    required = bool(condition.get("required", False))
    critical = bool(condition.get("critical", False))
    decision_on_fail = condition.get("decision_on_fail", "HOLD")
    accepted = set(condition.get("accepted_status") or [])
    max_age_days = condition.get("max_age_days")

    result: dict[str, Any] = {
        "name": condition["name"],
        "artifact": artifact_rel,
        "required": required,
        "critical": critical,
        "decision_on_fail": decision_on_fail,
        "status_observed": None,
        "issues": [],
        "passed": True,
    }

    if not artifact_path.exists():
        if required:
            result["passed"] = False
            result["issues"].append("missing")
        return result

    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        result["passed"] = False
        result["issues"].append(f"unparseable: {exc!s}")
        return result

    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    status = payload.get("status") or summary.get("status")
    result["status_observed"] = status
    if accepted and status not in accepted:
        result["passed"] = False
        result["issues"].append(f"status {status!r} not in {sorted(accepted)!r}")

    for threshold in condition.get("metric_thresholds") or []:
        metric_path = threshold.get("path") or []
        actual = _dig(payload, metric_path)
        if not _compare(actual, threshold.get("op"), threshold.get("value")):
            result["passed"] = False
            result["issues"].append(
                f"metric {'.'.join(str(p) for p in metric_path)}={actual!r} "
                f"failed {threshold.get('op')} {threshold.get('value')!r}"
            )

    if max_age_days is not None:
        ts = _parse_timestamp(payload.get("generated_at"))
        if ts is not None:
            age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
            if age_days > max_age_days:
                result["passed"] = False
                result["issues"].append(f"stale: {age_days:.1f}d > max_age_days={max_age_days}")

    return result


def _decision_from_results(results: list[dict[str, Any]]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    decision = "PROMOTE"
    for row in results:
        if row["passed"]:
            continue
        on_fail = row["decision_on_fail"]
        critical = row["critical"]
        reason = f"{row['name']}: {row['issues']!r}"
        reasons.append(reason)
        if on_fail == "REJECT" and critical:
            decision = "REJECT"
        elif decision != "REJECT" and on_fail in {"REJECT", "HOLD"}:
            decision = "HOLD"
    return decision, reasons


def run_mle_promotion_gate(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_yaml(config_path)
    results = [_check_condition(c) for c in config.get("conditions") or []]
    decision, reasons = _decision_from_results(results)
    return {
        "schema_version": "mle_promotion_gate_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "reasons": reasons,
        "condition_count": len(results),
        "passing_count": sum(1 for r in results if r["passed"]),
        "failing_count": sum(1 for r in results if not r["passed"]),
        "conditions": results,
        "config_path": str(config_path.relative_to(ROOT)),
        "claim_boundary": config.get("claim_boundary", ""),
    }


def _print_human(report: dict[str, Any]) -> None:
    decision = report["decision"]
    print(f"MLE promotion gate: {decision}")
    print(f"  config: {report['config_path']}")
    print(f"  conditions: {report['passing_count']}/{report['condition_count']} passing")
    print()
    for row in report["conditions"]:
        mark = "OK  " if row["passed"] else ("REJ " if row["decision_on_fail"] == "REJECT" else "HOLD")
        status = row.get("status_observed") or "-"
        print(f"  [{mark}] {row['name']:<40} status={status}")
        for issue in row["issues"]:
            print(f"        - {issue}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NLCare MLE promotion gate")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    report = run_mle_promotion_gate(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_human(report)
        print(f"\n  wrote {args.output.relative_to(ROOT)}")
    # PROMOTE -> 0 ; HOLD -> 2 ; REJECT -> 1.
    if report["decision"] == "PROMOTE":
        return 0
    if report["decision"] == "REJECT":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
