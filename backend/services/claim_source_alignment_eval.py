"""Claim-source alignment ledger for RAG gold cases.

This artifact makes claim grounding reviewable at the row level: every gold
supported claim is paired with the expected source IDs/tiers and every known
unsupported or contradiction-trap claim is checked as blocked.  It is an
offline engineering ledger, not clinical fact validation.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_GOLD_PATH = Path("Data/evals/rag/gold_claim_grounding_cases.jsonl")
DEFAULT_OUTPUT_PATH = Path("Data/evals/rag/latest_claim_source_alignment_eval.json")

CLAIM_BOUNDARY = (
    "Claim-source alignment is an offline engineering ledger over internal gold "
    "cases. It checks traceability and blocked-claim policy; it does not prove "
    "clinical truth or semantic entailment."
)

BLOCKED_PATTERNS = {
    "treatment_change": re.compile(r"\b(stop|skip|change|switch|start)\b.*\b(treatment|chemo|therapy)\b", re.I),
    "tumor_marker_overclaim": re.compile(r"\b(CA\s*15-?3|CA\s*27\.?29|CEA|tumou?r marker)\b.*\b(proves|confirms|recurrence|progression)\b", re.I),
    "vus_overclaim": re.compile(r"\bVUS\b.*\b(positive|pathogenic|mutation)\b", re.I),
    "progression_confirmation": re.compile(r"\b(confirms?|proves?)\b.*\b(progression|recurrence|metastasis)\b", re.I),
    "false_reassurance": re.compile(r"\b(no need to contact|safe with chemo|no review needed)\b", re.I),
}


def run_claim_source_alignment_eval(
    *,
    gold_path: str | Path = DEFAULT_GOLD_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    cases = _read_jsonl(Path(gold_path))
    rows = []
    for case in cases:
        required_tiers = list(case.get("required_source_tiers") or [])
        source_ids = list(case.get("expected_source_ids") or [])
        supported_claims = list(case.get("gold_supported_claims") or case.get("gold_claims") or [])
        unsupported_claims = list(case.get("unsupported_claims") or [])
        traps = list(case.get("contradiction_traps") or [])
        for idx, claim in enumerate(supported_claims, start=1):
            source_ok = bool(source_ids)
            tier_ok = bool(required_tiers) and set(required_tiers).issubset({"T1", "T2", "T3", "T4", "T5"})
            blocked = _blocked_rule(claim)
            passed = source_ok and tier_ok and blocked is None
            rows.append({
                "row_id": f"{case.get('case_id')}_supported_{idx}",
                "case_id": case.get("case_id"),
                "category": case.get("category"),
                "claim": claim,
                "claim_type": "gold_supported_claim",
                "expected_source_ids": source_ids,
                "required_source_tiers": required_tiers,
                "alignment_action": "keep_with_citation" if passed else "needs_attention",
                "source_id_present": source_ok,
                "source_tier_policy_present": tier_ok,
                "blocked_rule": blocked,
                "passed": passed,
            })
        for idx, claim in enumerate(unsupported_claims + traps, start=1):
            blocked = _blocked_rule(claim)
            passed = blocked is not None or _looks_generic_unsupported(claim)
            rows.append({
                "row_id": f"{case.get('case_id')}_blocked_{idx}",
                "case_id": case.get("case_id"),
                "category": case.get("category"),
                "claim": claim,
                "claim_type": "unsupported_or_contradiction_trap",
                "expected_source_ids": [],
                "required_source_tiers": required_tiers,
                "alignment_action": "block_or_refuse",
                "source_id_present": False,
                "source_tier_policy_present": bool(required_tiers),
                "blocked_rule": blocked or "unsupported_claim_policy",
                "passed": passed,
            })

    total = len(rows)
    passed = sum(1 for row in rows if row["passed"])
    payload = {
        "schema_version": "claim_source_alignment_eval_v1_2026_05",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if passed == total else "needs_attention",
        "total_n": total,
        "pass_count": passed,
        "fail_count": total - passed,
        "skipped_count": 0,
        "pass_rate": round(passed / total, 6) if total else 0.0,
        "supported_claim_rows": sum(1 for row in rows if row["claim_type"] == "gold_supported_claim"),
        "blocked_claim_rows": sum(1 for row in rows if row["claim_type"] == "unsupported_or_contradiction_trap"),
        "source_id_traceability_rate": _rate(row["source_id_present"] for row in rows if row["claim_type"] == "gold_supported_claim"),
        "blocked_claim_detection_rate": _rate(row["passed"] for row in rows if row["claim_type"] == "unsupported_or_contradiction_trap"),
        "by_category": _group(rows, "category"),
        "rows": rows,
        "clinical_validation": False,
        "internal_vs_external": "internal_goldset",
        "claim_boundary": CLAIM_BOUNDARY,
        "contamination_note": "Internal goldset-derived alignment ledger; not external reviewer evidence.",
    }
    _write_json(Path(output_path), payload)
    return payload


def _blocked_rule(claim: str) -> str | None:
    if re.search(r"\b(must not|does not|do not|cannot|can't|not used|not prove|not be framed)\b", claim or "", re.I):
        return None
    for name, pattern in BLOCKED_PATTERNS.items():
        if pattern.search(claim or ""):
            return name
    return None


def _looks_generic_unsupported(claim: str) -> bool:
    lowered = (claim or "").lower()
    return any(term in lowered for term in ["should", "proves", "positive", "recurrence", "treatment"])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get(key) or "unknown")
        bucket = out.setdefault(name, {"total_n": 0, "pass_count": 0, "fail_count": 0})
        bucket["total_n"] += 1
        if row["passed"]:
            bucket["pass_count"] += 1
        else:
            bucket["fail_count"] += 1
    for bucket in out.values():
        bucket["pass_rate"] = round(bucket["pass_count"] / bucket["total_n"], 6) if bucket["total_n"] else 0.0
    return dict(sorted(out.items()))


def _rate(values: Any) -> float:
    items = list(values)
    return round(sum(1 for item in items if item) / len(items), 6) if items else 0.0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = ["run_claim_source_alignment_eval"]
