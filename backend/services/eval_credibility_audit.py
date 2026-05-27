"""Audit internal eval artifacts for provenance and overclaim risk.

This does not score model quality.  It checks whether benchmark artifacts make
their limits visible: n-size, pass/fail counts, tuning/contamination metadata,
claim boundaries, and external-author status.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config" / "release_gate_thresholds.yaml"
DEFAULT_OUTPUT_PATH = ROOT / "Data/evals/governance/latest_eval_credibility_audit.json"


def run_eval_credibility_audit(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    resolved_config = _resolve_config_path(Path(config_path))
    artifact_base = ROOT if resolved_config == DEFAULT_CONFIG_PATH else resolved_config.parent
    config = _load_yaml(resolved_config)
    entries = config.get("artifacts") or []
    rows = []
    for entry in entries:
        rel_path = str(entry.get("path") or "")
        artifact_path = artifact_base / rel_path
        payload = _load_json(artifact_path)
        if payload is None:
            rows.append({
                "artifact_path": rel_path,
                "exists": False,
                "required": bool(entry.get("required", True)),
                "issues": ["missing_artifact"],
                "credibility_risk": "high" if entry.get("required", True) else "medium",
            })
            continue

        total_n = _find_first_number(payload, ("total_n", "case_count", "patient_count", "source_count", "dataset_count"))
        pass_count = _find_first_number(payload, ("pass_count", "passed_count", "checks_passed"))
        fail_count = _find_first_number(payload, ("fail_count", "failed_count", "checks_failed", "hard_failures"))
        skipped_count = _find_first_number(payload, ("skipped_count", "skip_count"))
        pass_rate = _find_first_number(payload, ("pass_rate", "safe_answer_rate", "source_tier_correctness"))
        external = _contains_value(payload, {"external", "external_author", "external_authored"})
        tuning_metadata = _has_key_recursive(payload, "was_used_for_tuning")
        contamination = _has_key_recursive(payload, "contamination_note")
        claim_boundary = _has_key_recursive(payload, "claim_boundary")
        clinical_false = _contains_pair(payload, "clinical_validation", False)
        provenance = _has_any_key_recursive(payload, {"authored_by", "authored_date", "case_source"})
        perfect_score = isinstance(pass_rate, (int, float)) and float(pass_rate) >= 1.0

        issues: list[str] = []
        if total_n is None:
            issues.append("missing_total_n_or_case_count")
        if pass_count is None and fail_count is None:
            issues.append("missing_pass_fail_counts")
        if not provenance:
            issues.append("missing_case_or_artifact_provenance")
        if not contamination:
            issues.append("missing_contamination_disclosure")
        if not claim_boundary:
            issues.append("missing_claim_boundary")
        if not clinical_false:
            issues.append("missing_explicit_clinical_validation_false")
        if perfect_score and not external:
            issues.append("perfect_internal_score_requires_caution")

        rows.append({
            "artifact_path": rel_path,
            "exists": True,
            "required": bool(entry.get("required", True)),
            "status": payload.get("status") or payload.get("overall_status"),
            "total_n": total_n,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "skipped_count": skipped_count,
            "pass_rate": pass_rate,
            "external_authored_detected": external,
            "has_tuning_metadata": tuning_metadata,
            "has_contamination_disclosure": contamination,
            "has_claim_boundary": claim_boundary,
            "clinical_validation_false_detected": clinical_false,
            "has_provenance_metadata": provenance,
            "perfect_internal_score": perfect_score and not external,
            "issues": issues,
            "credibility_risk": _risk_for(issues, required=bool(entry.get("required", True))),
        })

    summary = _summarize(rows)
    payload = {
        "schema_version": "eval_credibility_audit_v1_2026_05",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "acceptable",
        "summary": summary,
        "rows": rows,
        "clinical_validation": False,
        "claim_boundary": (
            "This audit improves benchmark transparency only. It does not make "
            "internal evals externally authored or clinically validated."
        ),
        "recommended_next_step": (
            "Keep this artifact as an honesty layer; the next credibility jump "
            "still requires external-author cases or expert review."
        ),
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    resolved = _resolve_config_path(path)
    return yaml.safe_load(resolved.read_text(encoding="utf-8"))


def _resolve_config_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    existing = [row for row in rows if row.get("exists")]
    total = len(rows)
    issue_rows = [row for row in existing if row.get("issues")]
    return {
        "artifact_count": total,
        "existing_artifact_count": len(existing),
        "missing_artifact_count": total - len(existing),
        "external_authored_artifact_count": sum(1 for row in existing if row.get("external_authored_detected")),
        "perfect_internal_score_count": sum(1 for row in existing if row.get("perfect_internal_score")),
        "n_size_metadata_rate": _coverage(existing, lambda row: row.get("total_n") is not None),
        "pass_fail_metadata_rate": _coverage(existing, lambda row: row.get("pass_count") is not None or row.get("fail_count") is not None),
        "provenance_metadata_rate": _coverage(existing, lambda row: bool(row.get("has_provenance_metadata"))),
        "contamination_disclosure_rate": _coverage(existing, lambda row: bool(row.get("has_contamination_disclosure"))),
        "claim_boundary_rate": _coverage(existing, lambda row: bool(row.get("has_claim_boundary"))),
        "clinical_validation_false_rate": _coverage(existing, lambda row: bool(row.get("clinical_validation_false_detected"))),
        "issue_artifact_count": len(issue_rows),
        "high_risk_artifact_count": sum(1 for row in rows if row.get("credibility_risk") == "high"),
        "top_issues": _top_issues(rows),
    }


def _coverage(rows: list[dict[str, Any]], predicate: Any) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if predicate(row)) / len(rows), 6)


def _top_issues(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for issue in row.get("issues") or []:
            counts[issue] = counts.get(issue, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8])


def _risk_for(issues: list[str], *, required: bool) -> str:
    if "missing_artifact" in issues:
        return "high" if required else "medium"
    if required and len(issues) >= 4:
        return "high"
    if len(issues) >= 3:
        return "medium"
    if issues:
        return "low"
    return "minimal"


def _find_first_number(payload: Any, keys: tuple[str, ...]) -> int | float | None:
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
        for value in payload.values():
            found = _find_first_number(value, keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_first_number(value, keys)
            if found is not None:
                return found
    return None


def _has_key_recursive(payload: Any, key: str) -> bool:
    return _has_any_key_recursive(payload, {key})


def _has_any_key_recursive(payload: Any, keys: set[str]) -> bool:
    if isinstance(payload, dict):
        if any(key in payload for key in keys):
            return True
        return any(_has_any_key_recursive(value, keys) for value in payload.values())
    if isinstance(payload, list):
        return any(_has_any_key_recursive(value, keys) for value in payload)
    return False


def _contains_pair(payload: Any, key: str, expected: Any) -> bool:
    if isinstance(payload, dict):
        if payload.get(key) == expected:
            return True
        return any(_contains_pair(value, key, expected) for value in payload.values())
    if isinstance(payload, list):
        return any(_contains_pair(value, key, expected) for value in payload)
    return False


def _contains_value(payload: Any, values: set[str]) -> bool:
    if isinstance(payload, str):
        return payload.lower() in values
    if isinstance(payload, dict):
        return any(_contains_value(value, values) for value in payload.values())
    if isinstance(payload, list):
        return any(_contains_value(value, values) for value in payload)
    return False


__all__ = ["run_eval_credibility_audit"]
