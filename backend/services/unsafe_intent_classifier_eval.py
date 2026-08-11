"""Report generation for the unsafe-intent classifier."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


# The first near-boundary example for these families is intentionally safe. This
# metadata belongs to the eval contract rather than the runtime classifier so a
# safe process or education question cannot be relabeled by classifier output.
_FIRST_NEAR_BOUNDARY_EXPECTED_SAFE = frozenset(
    {
        "privacy_pii",
        "cross_patient_exfiltration",
        "vus_misinterpretation",
        "tumor_marker_conclusion",
    }
)


def evaluate_classifier(
    *,
    output_path: str | Path,
    families: Iterable[Any],
    classify: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    family_rows = tuple(families)
    cases = _eval_cases(family_rows)
    rows = []
    for case in cases:
        result = classify(case["query"])
        passed = result["family"] == case["expected_family"] and (
            case["expect_unsafe"] == bool(result["is_unsafe"])
        )
        rows.append({**case, "actual": result, "passed": passed})

    total = len(rows)
    passed = sum(1 for row in rows if row["passed"])
    by_group: dict[str, dict[str, int | float]] = {}
    for row in rows:
        group = row["group"]
        by_group.setdefault(group, {"total_n": 0, "pass_count": 0, "fail_count": 0})
        by_group[group]["total_n"] += 1
        by_group[group]["pass_count"] += int(row["passed"])
        by_group[group]["fail_count"] += int(not row["passed"])
    for group in by_group.values():
        group["pass_rate"] = group["pass_count"] / group["total_n"] if group["total_n"] else 0.0

    payload = {
        "schema_version": "unsafe_intent_classifier_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if passed == total else "needs_attention",
        "total_n": total,
        "pass_count": passed,
        "fail_count": total - passed,
        "skipped_count": 0,
        "pass_rate": passed / total if total else 0.0,
        "by_group": by_group,
        "families": [
            {
                "family": family.family,
                "expected_route": family.expected_route,
                "positive_prototypes": list(family.positive_prototypes),
                "safe_negative_prototypes": list(family.safe_negative_prototypes),
                "near_boundary_examples": list(family.near_boundary_examples),
                "taglish_variants": list(family.taglish_variants),
                "over_refusal_risk_notes": family.over_refusal_risk_notes,
            }
            for family in family_rows
        ],
        "cases": rows,
        "claim_boundary": "Unsafe-intent classifier is an engineering routing aid, not clinical safety proof.",
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _eval_cases(families: tuple[Any, ...]) -> list[dict[str, Any]]:
    rows = []
    for family in families:
        for idx, query in enumerate(family.positive_prototypes[:2], start=1):
            rows.append({"case_id": f"{family.family}_pos_{idx}", "group": "unsafe_positive", "query": query, "expected_family": family.family, "expect_unsafe": True})
        for idx, query in enumerate(family.safe_negative_prototypes[:1], start=1):
            rows.append({"case_id": f"{family.family}_neg_{idx}", "group": "safe_negative", "query": query, "expected_family": "none", "expect_unsafe": False})
        if family.taglish_variants:
            rows.append({"case_id": f"{family.family}_taglish", "group": "taglish_variant", "query": family.taglish_variants[0], "expected_family": family.family, "expect_unsafe": True})
        if family.near_boundary_examples:
            expected_safe = family.family in _FIRST_NEAR_BOUNDARY_EXPECTED_SAFE
            rows.append(
                {
                    "case_id": f"{family.family}_near",
                    "group": "near_boundary_safe" if expected_safe else "near_boundary_unsafe",
                    "query": family.near_boundary_examples[0],
                    "expected_family": "none" if expected_safe else family.family,
                    "expect_unsafe": not expected_safe,
                }
            )
    return rows


__all__ = ["evaluate_classifier"]
