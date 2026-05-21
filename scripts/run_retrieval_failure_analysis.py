from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_retrieval_ablation_metrics import SOURCE_ID_ALIASES  # noqa: E402


INPUT_PATH = ROOT / "Data/evals/rag/latest_retrieval_goldset_eval.json"
OUTPUT_PATH = ROOT / "Data/evals/rag/latest_retrieval_failure_analysis.json"


def main() -> int:
    if not INPUT_PATH.exists():
        raise SystemExit("Run scripts/run_retrieval_goldset_eval.py first.")
    artifact = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    strategy = artifact.get("strategies", {}).get("hybrid_rrf_cross_encoder_source_governed") or {}
    cases = strategy.get("cases") or []
    rows = []
    category_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    ownership_counts: Counter[str] = Counter()

    for row in cases:
        if not row.get("unsupported_answer_proxy"):
            continue
        categories, owner, fixes = _classify_failure(row)
        for category in categories:
            category_counts[category] += 1
        intent = str(row.get("category") or row.get("expected_intent") or "unknown")
        intent_counts[intent] += 1
        ownership_counts[owner] += 1
        rows.append({
            "case_id": row.get("case_id"),
            "expected_source_ids": row.get("expected_source_ids"),
            "retrieved_source_ids": row.get("retrieved_source_ids"),
            "failure_categories": categories,
            "failure_owner": owner,
            "suggested_generalized_fixes": fixes,
            "notes": _note(row, categories),
        })

    payload = {
        "schema_version": "retrieval_failure_analysis_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "needs_attention" if rows else "strong",
        "total_n": len(cases),
        "failed_n": len(rows),
        "failure_categories": dict(sorted(category_counts.items())),
        "per_intent_failure_counts": dict(sorted(intent_counts.items())),
        "failure_owner_counts": dict(sorted(ownership_counts.items())),
        "suggested_generalized_fixes": _global_fixes(category_counts),
        "cases": rows,
        "claim_boundary": (
            "Retrieval failure analysis is engineering diagnostics only. It does not prove "
            "clinical answer quality or real-world retrieval reliability."
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "total_n": payload["total_n"],
        "failed_n": payload["failed_n"],
        "failure_categories": payload["failure_categories"],
    }, indent=2))
    return 0


def _classify_failure(row: dict[str, Any]) -> tuple[list[str], str, list[str]]:
    expected = {str(value).strip().lower() for value in row.get("expected_source_ids") or []}
    retrieved = {str(value).strip().lower() for value in row.get("retrieved_source_ids") or []}
    expanded_expected = _expand(expected)
    expanded_retrieved = _expand(retrieved)
    categories: list[str] = []
    fixes: list[str] = []

    if expected & SOURCE_ID_ALIASES.keys():
        categories.append("gold_source_id_alias_or_metadata_normalization")
        fixes.append("Normalize gold source aliases against current chunk id/title/parent_id/source_name fields.")
    if expanded_expected & expanded_retrieved:
        categories.append("relevant_source_retrieved_but_not_matched_by_raw_gold_id")
        fixes.append("Report raw-id and alias-normalized metrics separately.")
        return categories, "goldset_design", fixes

    joined = " ".join(sorted(retrieved))
    if any(term in joined for term in ("clinician", "protocol")):
        categories.append("clinician_only_source_conflict")
        fixes.append("Keep clinician-only filtering before patient-facing generation.")
    if not retrieved:
        categories.append("retrieval_empty")
        fixes.append("Use source-tier-aware fallback to safety policy or insufficient-evidence templates.")
    else:
        categories.append("unsupported_context_selected")
        fixes.append("Improve intent-aware query rewrite and parent-child chunk recovery for this intent family.")

    if any(str(value).startswith(("retrieval_gold_claim_04",)) for value in [row.get("case_id")]):
        categories.append("chunking_or_heading_issue")
        fixes.append("Boost imaging response headings and parent chunks when MRI/CT/ultrasound terms appear.")
    if any(str(value).startswith(("retrieval_gold_claim_05", "retrieval_gold_claim_06")) for value in [row.get("case_id")]):
        categories.append("source_governance_or_intent_boost_gap")
        fixes.append("Boost genetics/VUS and tumor-marker boundary documents for matching high-risk intents.")
    if any(str(value).startswith(("retrieval_gold_claim_09", "retrieval_gold_claim_12")) for value in [row.get("case_id")]):
        categories.append("safety_policy_fallback_gap")
        fixes.append("For refusal/privacy/prognosis intents, include project safety policy as fallback context.")

    owner = "retrieval" if "unsupported_context_selected" in categories else "metadata"
    return sorted(set(categories)), owner, sorted(set(fixes))


def _expand(values: set[str]) -> set[str]:
    expanded = set(values)
    for value in values:
        expanded |= SOURCE_ID_ALIASES.get(value, set())
    return expanded


def _note(row: dict[str, Any], categories: list[str]) -> str:
    if "relevant_source_retrieved_but_not_matched_by_raw_gold_id" in categories:
        return "The current KB appears to retrieve an equivalent source, but the gold ID uses an older alias."
    return "No expected source or alias appeared in top-10 retrieval for this case."


def _global_fixes(counts: Counter[str]) -> list[str]:
    fixes = []
    if counts.get("gold_source_id_alias_or_metadata_normalization"):
        fixes.append("Maintain a source-alias map and expose alias-normalized retrieval metrics.")
    if counts.get("unsupported_context_selected"):
        fixes.append("Add intent-specific metadata boosts for refusal, genetics/VUS, tumor-marker, imaging, and privacy routes.")
    if counts.get("safety_policy_fallback_gap"):
        fixes.append("Add a source-governed safety-policy fallback for refusal/privacy/prognosis route families.")
    if counts.get("chunking_or_heading_issue"):
        fixes.append("Use heading-aware parent-child recovery for imaging response sections.")
    return fixes or ["Keep monitoring retrieval failures as the KB evolves."]


if __name__ == "__main__":
    raise SystemExit(main())
