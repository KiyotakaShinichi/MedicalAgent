"""NLCare release gate.

Aggregates the existing audit artifacts under ``Data/evals/{models,rag,
safety,medical}/`` and fails if any required artifact is missing, stale,
or below its accepted status.

Thresholds live in ``config/release_gate_thresholds.yaml`` so the policy is
visible and version-controlled in one place.

Exit codes
~~~~~~~~~~
  0  every required artifact present, fresh, and at an accepted status
  1  one or more required artifacts missing / stale / failed status

Usage
~~~~~
    python scripts/run_release_gate.py
    python scripts/run_release_gate.py --config config/release_gate_thresholds.yaml
    python scripts/run_release_gate.py --json   # machine-readable summary

The gate is wired into ``make ship`` as the final step.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "release_gate_thresholds.yaml"
LEGACY_CONFIG = ROOT / "scripts" / "release_gate_config.json"


def _load_config(config_path: Path) -> dict[str, Any]:
    path = config_path if config_path.is_absolute() else ROOT / config_path
    if not path.exists() and path == DEFAULT_CONFIG and LEGACY_CONFIG.exists():
        path = LEGACY_CONFIG
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency is in requirements.txt
            raise RuntimeError("PyYAML is required for YAML release-gate configs.") from exc
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Release-gate config must be an object: {path}")
    payload["_resolved_config_path"] = path
    return payload


def _parse_timestamp(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        # Accept both ``...+00:00`` and ``...Z`` variants.
        cleaned = value.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except Exception:
        return None


def _extract_generated_at(payload: dict[str, Any]) -> datetime | None:
    for key in ("generated_at", "generated_at_iso", "as_of", "timestamp"):
        ts = _parse_timestamp(payload.get(key))
        if ts is not None:
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts
    return None


def _check_artifact(entry: dict[str, Any]) -> dict[str, Any]:
    rel_path = entry["path"]
    abs_path = ROOT / rel_path
    required = bool(entry.get("required", True))
    accepted = set(entry.get("accepted_status") or [])
    max_age_days = entry.get("max_age_days")

    result: dict[str, Any] = {
        "path": rel_path,
        "required": required,
        "exists": abs_path.exists(),
        "status": None,
        "age_days": None,
        "issues": [],
    }

    if not abs_path.exists():
        if required:
            result["issues"].append("missing")
        return result

    try:
        payload = json.loads(abs_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — broad catch by design
        result["issues"].append(f"unparseable: {exc!s}")
        return result

    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    status = payload.get("status") or summary.get("status")
    result["status"] = status
    if accepted and status not in accepted:
        result["issues"].append(f"status {status!r} not in accepted {sorted(accepted)!r}")

    for threshold in entry.get("metric_thresholds") or []:
        metric_path = threshold.get("path") or []
        actual = _dig(payload, metric_path)
        ok = _compare(actual, threshold.get("op"), threshold.get("value"))
        if not ok:
            result["issues"].append(
                f"metric {'.'.join(str(part) for part in metric_path)} "
                f"={actual!r} failed {threshold.get('op')} {threshold.get('value')!r}"
            )

    ts = _extract_generated_at(payload)
    if ts is not None:
        age = datetime.now(timezone.utc) - ts
        result["age_days"] = round(age.total_seconds() / 86400, 2)
        if max_age_days is not None and result["age_days"] > max_age_days:
            result["issues"].append(
                f"stale: {result['age_days']:.1f}d > max_age_days={max_age_days}"
            )
    return result


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
    raise ValueError(f"Unsupported release-gate comparator: {op!r}")


def run_release_gate(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_config(config_path)
    resolved_config = config.get("_resolved_config_path")
    if isinstance(resolved_config, Path):
        config_label = str(resolved_config.relative_to(ROOT))
    else:
        config_label = str(config_path)
    artifacts = config.get("artifacts") or []
    rows = [_check_artifact(entry) for entry in artifacts]

    failures = [r for r in rows if r["required"] and r["issues"]]
    overall_status = "passed" if not failures else "failed"
    return {
        "schema_version": "release_gate_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": overall_status,
        "config_path": config_label,
        "artifact_count": len(rows),
        "failure_count": len(failures),
        "artifacts": rows,
    }


def _print_human(report: dict[str, Any]) -> None:
    print(f"NLCare release gate: {report['status'].upper()}")
    print(f"  config: {report['config_path']}")
    print(f"  artifacts checked: {report['artifact_count']}")
    print(f"  failures: {report['failure_count']}")
    print()
    for row in report["artifacts"]:
        mark = "OK  " if not row["issues"] else "FAIL" if row["required"] else "warn"
        age = f"{row['age_days']}d" if row["age_days"] is not None else "?"
        status = row["status"] or "(no status field)"
        print(f"  [{mark}] {row['path']}  status={status}  age={age}")
        for issue in row["issues"]:
            print(f"        - {issue}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NLCare release gate")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    report = run_release_gate(args.config)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_human(report)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
