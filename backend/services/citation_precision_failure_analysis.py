"""Citation-precision failure analysis.

Reads the most recent baseline comparison + failures artifacts and
classifies every full-stack failure that has ``low_citation_precision``
into one of 12 owner-tagged categories. Emits a JSON report with
counts, per-intent counts, representative examples, and a generalised-
fix proposal per category.

The classifier is **heuristic and generic** — it inspects retrieved
IDs / titles, the case's expected source IDs, the rewritten-query
trail (where available), and the source-tier filtering trail. It does
NOT modify ranking, the goldset, or any thresholds.

Output: ``Data/evals/rag/latest_citation_precision_failure_analysis.json``
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COMPARISON_PATH = Path("Data/evals/rag/latest_rag_baseline_comparison.json")
FAILURES_PATH = Path("Data/evals/rag/latest_rag_baseline_failures.json")
OUTPUT_PATH = Path("Data/evals/rag/latest_citation_precision_failure_analysis.json")

FULL_STACK_CONFIG_ID = "hybrid_rrf_query_rewrite_parent_child_source_tier"


# 12 failure categories, in the order the brief listed them.
# Each carries an owner tag (where the generalised fix lives).
CATEGORIES: tuple[dict[str, str], ...] = (
    {"key": "irrelevant_sibling_chunk_included",       "owner": "context_pruning"},
    {"key": "parent_child_expansion_too_broad",        "owner": "context_pruning"},
    {"key": "query_rewrite_drift",                     "owner": "retrieval_ranking"},
    {"key": "source_alias_or_metadata_mismatch",       "owner": "metadata"},
    {"key": "source_tier_filter_removed_relevant",     "owner": "metadata"},
    {"key": "bm25_lexical_distractor",                 "owner": "retrieval_ranking"},
    {"key": "dense_semantic_distractor",               "owner": "retrieval_ranking"},
    {"key": "duplicated_near_equivalent_chunk",        "owner": "context_pruning"},
    {"key": "low_value_safety_policy_chunk_over_selected", "owner": "citation_assembly"},
    {"key": "expected_gold_source_too_narrow",         "owner": "goldset_design"},
    {"key": "insufficient_top_k_pruning",              "owner": "context_pruning"},
    {"key": "citation_assembly_includes_non_supporting", "owner": "citation_assembly"},
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2}


@dataclass
class _CaseFailure:
    case_id: str
    intent: str
    refusal_route: bool
    query: str
    expected: list[str]
    retrieved: list[str]
    citation_precision: float
    failure_reasons: list[str] = field(default_factory=list)


def _load_failures() -> list[_CaseFailure]:
    if not FAILURES_PATH.exists():
        return []
    data = json.loads(FAILURES_PATH.read_text(encoding="utf-8"))
    rows: list[_CaseFailure] = []
    for f in data.get("failures") or []:
        if f.get("configuration") != FULL_STACK_CONFIG_ID:
            continue
        reasons = list(f.get("failure_reasons") or [])
        if "low_citation_precision" not in reasons:
            continue
        rows.append(
            _CaseFailure(
                case_id=str(f.get("case_id") or "?"),
                intent=str(f.get("expected_intent") or ""),
                refusal_route=bool(f.get("expected_refusal_or_insufficient_evidence")),
                query=str(f.get("query") or ""),
                expected=[str(s) for s in (f.get("expected_source_ids") or [])],
                retrieved=[str(s) for s in (f.get("retrieved_source_ids") or [])],
                citation_precision=float(f.get("citation_precision") or 0.0),
                failure_reasons=reasons,
            )
        )
    return rows


def _classify(failure: _CaseFailure) -> list[str]:
    """Heuristic owner tagging.

    A single failure can sit in multiple categories — we return the set
    of categories that fire. If none fire, we return
    ['citation_assembly_includes_non_supporting'] as the fallback
    (the most generic "the cited context was junk" bucket).
    """
    hits: list[str] = []
    expected_norm = {e.lower() for e in failure.expected}
    retrieved_norm = [r.lower() for r in failure.retrieved]
    query_toks = _tokens(failure.query)

    # 1. irrelevant sibling chunk — a retrieved chunk shares a parent
    #    prefix with one that IS expected but isn't itself expected.
    if len(retrieved_norm) >= 2:
        sib_count = sum(
            1 for r in retrieved_norm
            if r not in expected_norm and any(
                r != e and (r[:6] == e[:6] or e in r or r in e)
                for e in expected_norm
            )
        )
        if sib_count >= 1:
            hits.append("irrelevant_sibling_chunk_included")

    # 2. parent-child expansion too broad — many distinct retrieved
    #    chunks but very few matched.
    matched = sum(1 for r in retrieved_norm if any(e in r or r in e for e in expected_norm))
    if len(retrieved_norm) >= 5 and matched <= 1:
        hits.append("parent_child_expansion_too_broad")

    # 3. query rewrite drift — query tokens have weak overlap with the
    #    expected source IDs' tokens.
    expected_toks: set[str] = set()
    for e in expected_norm:
        expected_toks |= _tokens(e)
    if query_toks and expected_toks and not (query_toks & expected_toks):
        hits.append("query_rewrite_drift")

    # 4. source alias / metadata mismatch — at least one retrieved
    #    chunk looks like a hashed ID (16-char hex) while the expected
    #    list is human-readable canonicals.
    hashed_re = re.compile(r"^[0-9a-f]{12,40}$")
    if any(hashed_re.fullmatch(r) for r in retrieved_norm) and any(
        not hashed_re.fullmatch(e) for e in expected_norm
    ):
        hits.append("source_alias_or_metadata_mismatch")

    # 5. source-tier filter removed relevant — retrieved is empty AND
    #    refusal_route is False (tier filter dropped everything for a
    #    patient-facing query).
    if not failure.retrieved and not failure.refusal_route:
        hits.append("source_tier_filter_removed_relevant")

    # 6/7. lexical vs semantic distractor — without ranking access we
    #      can't disambiguate which retriever hurt.  Use a conservative
    #      proxy: if a retrieved id contains common-noise tokens
    #      ("policy", "safety", "boundary") AND the case is education,
    #      flag as semantic distractor; otherwise lexical distractor.
    if matched == 0 and not failure.refusal_route:
        looks_semantic = any(
            tok in (r if isinstance(r, str) else "")
            for r in retrieved_norm
            for tok in ("safety", "boundary", "policy")
        )
        hits.append("dense_semantic_distractor" if looks_semantic else "bm25_lexical_distractor")

    # 8. duplicated near-equivalent — same parent_id appears more than once.
    if len(retrieved_norm) >= 2 and len(set(retrieved_norm)) < len(retrieved_norm):
        hits.append("duplicated_near_equivalent_chunk")

    # 9. low-value safety-policy chunk over-selected — education intent
    #    with retrieved containing safety/policy phrases.
    if not failure.refusal_route and any(
        tok in r for r in retrieved_norm for tok in ("safety", "policy", "boundary")
    ):
        hits.append("low_value_safety_policy_chunk_over_selected")

    # 10. expected gold source too narrow — only one expected ID and
    #     citation_precision is very low.
    if len(expected_norm) <= 1 and failure.citation_precision <= 0.2:
        hits.append("expected_gold_source_too_narrow")

    # 11. insufficient top-k pruning — citation_precision = 0 (top-5
    #     had nothing).
    if failure.citation_precision == 0.0:
        hits.append("insufficient_top_k_pruning")

    # 12. citation assembly includes non-supporting context — catches
    #     anything we haven't otherwise labeled.
    if not hits:
        hits.append("citation_assembly_includes_non_supporting")
    return hits


# Generalised-fix proposals per category.  Each fix is **generic** —
# no goldset case_ids, no hardcoded source IDs.
_GENERALIZED_FIXES: dict[str, str] = {
    "irrelevant_sibling_chunk_included": (
        "Apply citation_context_pruner with per-parent_id dedup and a marginal-coverage gate "
        "on sibling chunks (new lexical tokens vs. already-covered set)."
    ),
    "parent_child_expansion_too_broad": (
        "Cap parent-child expansion to seeds whose own retrieval_score is above the median, "
        "then re-prune via citation_context_pruner before citation assembly."
    ),
    "query_rewrite_drift": (
        "Add a query-rewrite-drift detector that vetoes the rewrite when its token overlap "
        "with the original query falls below a configurable floor; defaults preserved."
    ),
    "source_alias_or_metadata_mismatch": (
        "Run scripts/run_source_alias_coverage.py before the next baseline run; promote "
        "diagnostic-proposed aliases per ADR 0009."
    ),
    "source_tier_filter_removed_relevant": (
        "Surface the dropped-chunk reason in tier_filter trace; surface insufficient_evidence "
        "rather than silently emptying the context window."
    ),
    "bm25_lexical_distractor": (
        "Reweight BM25 contribution in the hybrid fusion when dense top-1 score is high; "
        "no change for BM25-only baselines."
    ),
    "dense_semantic_distractor": (
        "Penalise dense candidates whose chunk title/topic does not overlap the query intent "
        "(intent-tag bonus already in citation_context_pruner)."
    ),
    "duplicated_near_equivalent_chunk": (
        "First-pass dedup is already in citation_context_pruner; ensure parent_key uses "
        "parent_id || source_name || id consistently."
    ),
    "low_value_safety_policy_chunk_over_selected": (
        "citation_context_pruner already penalises boundary sources in non-refusal routes; "
        "verify the penalty is large enough by re-running with -0.15 instead of -0.10."
    ),
    "expected_gold_source_too_narrow": (
        "Goldset design: when promoting a case past peer review, allow up to 2 acceptable "
        "expected_source_ids rather than 1 to reduce citation_precision noise."
    ),
    "insufficient_top_k_pruning": (
        "Citation context window (CITED_CONTEXT_K) is currently 5; the pruner shrinks "
        "post-governance to 10 then citation_precision is computed on top-5.  Re-check "
        "whether CITED_CONTEXT_K=3 reduces unsupported_context_rate without breaking "
        "refusal cases."
    ),
    "citation_assembly_includes_non_supporting": (
        "Tighten validate_claims's overlap threshold by 0.02 and re-run baseline; if "
        "citation_precision improves without unsafe_answer_rate increase, promote the "
        "threshold via the threshold-calibration sweep."
    ),
}


def build_report() -> dict[str, Any]:
    failures = _load_failures()
    cat_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    examples_by_cat: dict[str, list[dict[str, Any]]] = {}

    for failure in failures:
        categories = _classify(failure)
        for cat in categories:
            cat_counts[cat] += 1
            examples_by_cat.setdefault(cat, [])
            if len(examples_by_cat[cat]) < 3:
                examples_by_cat[cat].append({
                    "case_id": failure.case_id,
                    "expected_intent": failure.intent,
                    "query": failure.query,
                    "expected": failure.expected,
                    "retrieved": failure.retrieved[:5],
                    "citation_precision": failure.citation_precision,
                })
        intent_counts[failure.intent or "unknown"] += 1

    category_summary: list[dict[str, Any]] = []
    for cat in CATEGORIES:
        key = cat["key"]
        category_summary.append({
            "category": key,
            "owner": cat["owner"],
            "count": cat_counts.get(key, 0),
            "examples": examples_by_cat.get(key, []),
            "generalized_fix": _GENERALIZED_FIXES.get(key, ""),
        })

    status = (
        "needs_attention"
        if failures and any(c["count"] > 0 for c in category_summary)
        else "informational"
    )

    return {
        "schema_version": "1.0",
        "status": status,
        "label": "citation_precision_failure_analysis",
        "claim_boundary": (
            "Engineering signal only.  The classifier is heuristic and operates on the "
            "frozen internal goldset.  It does not establish clinical validity or external "
            "generalisation.  Generalised fixes proposed here are NOT auto-applied; each "
            "must be evaluated by re-running the baseline comparison and confirming the "
            "metric trade is acceptable."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": {
            "comparison": str(COMPARISON_PATH).replace("\\", "/"),
            "failures": str(FAILURES_PATH).replace("\\", "/"),
        },
        "configuration_under_analysis": FULL_STACK_CONFIG_ID,
        "total_n": len(failures),
        "failed_n": len(failures),
        "per_intent_failure_counts": dict(intent_counts),
        "category_counts": {key: cat_counts.get(key, 0) for key in (c["key"] for c in CATEGORIES)},
        "category_summary": category_summary,
        "contamination_note": (
            "Categories are inferred from in-sample failures.  Treat the fix proposals as "
            "hypotheses to test on a held-out goldset (docs/evals/no_read_rag_goldset_protocol.md)."
        ),
    }


def write_report(output_path: Path = OUTPUT_PATH) -> Path:
    report = build_report()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


__all__ = ["CATEGORIES", "OUTPUT_PATH", "build_report", "write_report"]
