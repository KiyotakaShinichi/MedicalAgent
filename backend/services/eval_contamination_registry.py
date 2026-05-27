"""Machine-readable registry for internal eval provenance and contamination."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = ROOT / "Data/evals/governance/latest_eval_contamination_registry.json"
EVAL_ROOTS = [ROOT / "Data/evals/rag", ROOT / "Data/evals/safety", ROOT / "Data/evals/agentic_tool_use"]


def run_eval_contamination_registry(output_path: str | Path = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    rows = []
    for root in EVAL_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.jsonl")):
            cases = _read_jsonl(path)
            if not cases:
                continue
            rows.append(_summarize_case_file(path, cases))
        for path in sorted(root.glob("latest_*.json")):
            payload = _read_json(path)
            if payload is not None:
                rows.append(_summarize_artifact(path, payload))

    total = len(rows)
    tuned = sum(1 for row in rows if row["any_used_for_tuning"])
    external = sum(1 for row in rows if row["external_authored_case_count"] > 0)
    frozen = sum(1 for row in rows if row["frozen_or_holdout"])
    payload = {
        "schema_version": "eval_contamination_registry_v1_2026_05",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "acceptable",
        "total_n": total,
        "pass_count": total,
        "fail_count": 0,
        "skipped_count": 0,
        "registry_entry_count": total,
        "used_for_tuning_entry_count": tuned,
        "external_authored_entry_count": external,
        "frozen_or_holdout_entry_count": frozen,
        "rows": rows,
        "clinical_validation": False,
        "claim_boundary": (
            "This registry documents internal eval provenance and contamination "
            "risk. It does not make internal cases external or clinically valid."
        ),
        "next_step": "Add truly external-authored cases and record reviewer role/date before treating any eval as independent evidence.",
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _summarize_case_file(path: Path, cases: list[dict[str, Any]]) -> dict[str, Any]:
    used = [case for case in cases if bool(case.get("was_used_for_tuning"))]
    external = [
        case for case in cases
        if str(case.get("internal_vs_external_authored") or case.get("internal_vs_external") or "").lower().startswith("external")
    ]
    contamination = [
        case for case in cases
        if case.get("contamination_note") or case.get("contamination_notes") or case.get("contamination_disclosure")
    ]
    return {
        "artifact_path": _rel(path),
        "artifact_type": "case_bank",
        "case_count": len(cases),
        "authored_by_values": sorted({str(case.get("authored_by") or "unknown") for case in cases}),
        "authored_date_values": sorted({str(case.get("authored_date") or "unknown") for case in cases}),
        "any_used_for_tuning": bool(used),
        "used_for_tuning_count": len(used),
        "external_authored_case_count": len(external),
        "contamination_disclosure_rate": round(len(contamination) / len(cases), 6) if cases else 0.0,
        "frozen_or_holdout": _looks_holdout(path.name),
        "recommended_use": "holdout_warning" if _looks_holdout(path.name) else "internal_regression",
        "credibility_note": _credibility_note(path.name, bool(used), len(external)),
    }


def _summarize_artifact(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows") or payload.get("cases") or payload.get("results") or []
    if not isinstance(rows, list):
        rows = []
    total = int(payload.get("total_n") or payload.get("summary", {}).get("case_count") or len(rows) or 0)
    used = _contains_pair(payload, "was_used_for_tuning", True)
    external = _contains_value(payload, {"external", "external_author", "external_authored"})
    return {
        "artifact_path": _rel(path),
        "artifact_type": "result_artifact",
        "case_count": total,
        "authored_by_values": sorted(_collect_values(payload, "authored_by")),
        "authored_date_values": sorted(_collect_values(payload, "authored_date")),
        "any_used_for_tuning": used,
        "used_for_tuning_count": _count_pair(payload, "was_used_for_tuning", True),
        "external_authored_case_count": 1 if external else 0,
        "contamination_disclosure_rate": 1.0 if _has_any_key(payload, {"contamination_note", "contamination_notes", "contamination_disclosure"}) else 0.0,
        "frozen_or_holdout": _looks_holdout(path.name),
        "recommended_use": "supporting_artifact",
        "credibility_note": _credibility_note(path.name, used, 1 if external else 0),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _looks_holdout(name: str) -> bool:
    lowered = name.lower()
    return any(term in lowered for term in ["holdout", "frozen", "external_author", "template"])


def _credibility_note(name: str, used_for_tuning: bool, external_count: int) -> str:
    if external_count > 0:
        return "Contains external-authored metadata; verify reviewer independence before using as external evidence."
    if used_for_tuning:
        return "Internal tuning/regression evidence; report separately from holdout or external-author results."
    if _looks_holdout(name):
        return "Internal holdout/template evidence; do not tune against this exact set without creating a replacement."
    return "Internal regression/supporting evidence."


def _collect_values(payload: Any, key: str) -> set[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        if key in payload:
            found.add(str(payload[key]))
        for value in payload.values():
            found.update(_collect_values(value, key))
    elif isinstance(payload, list):
        for value in payload:
            found.update(_collect_values(value, key))
    return found


def _contains_pair(payload: Any, key: str, expected: Any) -> bool:
    return _count_pair(payload, key, expected) > 0


def _count_pair(payload: Any, key: str, expected: Any) -> int:
    if isinstance(payload, dict):
        count = 1 if payload.get(key) == expected else 0
        return count + sum(_count_pair(value, key, expected) for value in payload.values())
    if isinstance(payload, list):
        return sum(_count_pair(value, key, expected) for value in payload)
    return 0


def _contains_value(payload: Any, values: set[str]) -> bool:
    if isinstance(payload, str):
        return payload.lower() in values
    if isinstance(payload, dict):
        return any(_contains_value(value, values) for value in payload.values())
    if isinstance(payload, list):
        return any(_contains_value(value, values) for value in payload)
    return False


def _has_any_key(payload: Any, keys: set[str]) -> bool:
    if isinstance(payload, dict):
        if any(key in payload for key in keys):
            return True
        return any(_has_any_key(value, keys) for value in payload.values())
    if isinstance(payload, list):
        return any(_has_any_key(value, keys) for value in payload)
    return False


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


__all__ = ["run_eval_contamination_registry"]
