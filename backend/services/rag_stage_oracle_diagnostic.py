"""Stage-wise retrieval oracle diagnostic.

For each frozen-goldset case, re-runs every retrieval stage in
isolation and records whether the expected gold source survives that
stage. Emits a single JSON artifact that tells us *where* source
grounding is being lost — candidate generation, query rewriting,
fusion ranking, parent-child expansion, source-tier filtering, or
the final citation window.

This module is **read-only**:

* It does not modify the goldset.
* It does not call any live patient-agent generation path.
* It does not change any retrieval ranking decision in
  ``backend/services/rag_baseline_comparison.py`` or in the live
  agent stack.
* It does not promote the citation-context pruner.

It re-uses the same retrieval primitives the baseline comparison
uses, so a divergence between this diagnostic and the baseline run is
itself evidence of a change in the retrieval surface.

Output: ``Data/evals/rag/latest_rag_stage_oracle_diagnostic.json``
"""
from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from backend.services.agent_query_rewriting import rewrite_and_decompose
from backend.services.agent_rag import _knowledge_snippets, knowledge_base_fingerprint
from backend.services.agent_retrieval import expand_parent_child_windows
from backend.services.rag_vector_index import search_hybrid_index
from backend.services.rag_baseline_comparison import (
    _apply_case_source_filter,
    _bm25_only_retrieval,
    _dedupe_rows,
    _expected_source_groups,
    _matched_expected_groups,
    _map_goldset_intent,
    _row_ids,
)


DEFAULT_GOLDSET_PATH = Path("Data/evals/rag/retrieval_goldset.jsonl")
DEFAULT_OUTPUT_PATH = Path("Data/evals/rag/latest_rag_stage_oracle_diagnostic.json")

CANDIDATE_LIMIT = 50
PARENT_CHILD_SEED_LIMIT = 20
EVAL_CONTEXT_K = 10
CITED_CONTEXT_K = 5

# Final-failure stage vocabulary, in escalation order. We attribute
# each failed case to the *earliest* stage at which the expected
# source disappears. "no_failure" means the case actually succeeds
# (an expected source survives all the way to the citation window).
FAILURE_STAGES: tuple[str, ...] = (
    "gold_source_missing_from_corpus",
    "candidate_generation_failure",
    "dense_failure",
    "sparse_failure",
    "rrf_ranking_failure",
    "query_rewrite_drift",
    "parent_child_expansion_noise",
    "source_filter_drop",
    "citation_window_drop",
    "metadata_alias_mismatch",
    "goldset_design_possible_issue",
    "no_failure",
)


# ─── Helpers ────────────────────────────────────────────────────────────


def _rows_hit_any_group(rows: Iterable[Mapping[str, Any]], expected_groups: list[set[str]]) -> bool:
    if not expected_groups:
        return False
    for row in rows:
        ids = _row_ids(row)
        if any(ids & group for group in expected_groups):
            return True
    return False


def _bm25_candidates(query: str, corpus: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return _bm25_only_retrieval(query, corpus, limit=limit)


def _hybrid_candidates(
    query: str,
    intent: str,
    corpus: list[dict[str, Any]],
    fingerprint: str,
    limit: int,
) -> list[dict[str, Any]]:
    return search_hybrid_index(
        query=query,
        corpus=corpus,
        intent=intent,
        knowledge_fingerprint=fingerprint,
        candidate_limit=limit,
    )


def _dense_only(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-rank the hybrid candidates by dense_score only.

    Mirrors how ``rag_baseline_comparison._retrieve_for_config`` builds
    its FAISS-dense-only configuration: it starts from the same hybrid
    candidate pool and sorts by ``dense_score``/``vector_score``.
    """
    return sorted(
        candidates,
        key=lambda r: float(r.get("dense_score") or r.get("vector_score") or 0.0),
        reverse=True,
    )


def _has_in_corpus(corpus: list[dict[str, Any]], expected_groups: list[set[str]]) -> bool:
    if not expected_groups:
        return False
    for row in corpus:
        ids = _row_ids(row)
        if any(ids & group for group in expected_groups):
            return True
    return False


def _looks_like_alias_mismatch(case: Mapping[str, Any]) -> bool:
    """Heuristic: case expects only human-readable canonicals AND the
    cited window failed despite candidate stages succeeding."""
    raw = [str(s).strip() for s in (case.get("expected_source_ids") or [])]
    if not raw:
        return False
    # Treat a value with at least one dash and no hex-only as "human-readable".
    human_readable = sum(1 for s in raw if "-" in s and not all(ch in "0123456789abcdef" for ch in s.replace("-", "").lower()))
    return human_readable >= max(1, len(raw) // 2)


# ─── Per-case oracle pass ───────────────────────────────────────────────


def _diagnose_case(
    case: Mapping[str, Any],
    corpus: list[dict[str, Any]],
    fingerprint: str,
) -> dict[str, Any]:
    query = str(case.get("user_query") or case.get("query") or "")
    raw_intent = str(case.get("expected_intent") or "education")
    intent = _map_goldset_intent(raw_intent)
    expected_groups = _expected_source_groups(case)

    # Stage 0: corpus
    corpus_has_expected = _has_in_corpus(corpus, expected_groups)

    # Stage 1a: BM25 candidates
    bm25 = _bm25_candidates(query, corpus, CANDIDATE_LIMIT)
    bm25_top_50 = _rows_hit_any_group(bm25, expected_groups)
    bm25_top_20 = _rows_hit_any_group(bm25[:20], expected_groups)
    bm25_top_10 = _rows_hit_any_group(bm25[:10], expected_groups)

    # Stage 1b: hybrid candidates pool (used by dense and hybrid configs)
    hybrid_pool = _hybrid_candidates(query, intent, corpus, fingerprint, CANDIDATE_LIMIT)
    dense_ranked = _dense_only(hybrid_pool)
    dense_top_50 = _rows_hit_any_group(dense_ranked, expected_groups)
    dense_top_20 = _rows_hit_any_group(dense_ranked[:20], expected_groups)
    dense_top_10 = _rows_hit_any_group(dense_ranked[:10], expected_groups)

    hybrid_ranked = sorted(
        hybrid_pool,
        key=lambda r: float(r.get("retrieval_score") or 0.0),
        reverse=True,
    )
    hybrid_top_50 = _rows_hit_any_group(hybrid_ranked, expected_groups)
    hybrid_top_20 = _rows_hit_any_group(hybrid_ranked[:20], expected_groups)
    hybrid_top_10 = _rows_hit_any_group(hybrid_ranked[:10], expected_groups)

    # Stage 2: query rewrite (run only once per case)
    rewritten = rewrite_and_decompose(query, intent)
    rewritten_query = str(rewritten.get("expanded_query") or query)
    query_rewrite_changed = rewritten_query.strip() != query.strip()

    hybrid_rewritten_pool: list[dict[str, Any]] = []
    if query_rewrite_changed:
        hybrid_rewritten_pool = _hybrid_candidates(
            rewritten_query, intent, corpus, fingerprint, CANDIDATE_LIMIT,
        )
    hybrid_rewritten_ranked = sorted(
        hybrid_rewritten_pool or hybrid_pool,
        key=lambda r: float(r.get("retrieval_score") or 0.0),
        reverse=True,
    )

    # query_rewrite_helped if rewrite surfaces an expected source that the
    # original-query hybrid did NOT have in its top-10.
    rewrite_top_10_hits = _rows_hit_any_group(hybrid_rewritten_ranked[:10], expected_groups)
    query_rewrite_helped = bool(
        query_rewrite_changed and rewrite_top_10_hits and not hybrid_top_10
    )
    query_rewrite_hurt = bool(
        query_rewrite_changed and (not rewrite_top_10_hits) and hybrid_top_10
    )

    # Stage 3: parent-child expansion seeded from hybrid (or rewritten hybrid)
    seed = (hybrid_rewritten_ranked or hybrid_ranked)[:PARENT_CHILD_SEED_LIMIT]
    expanded = expand_parent_child_windows(seed)
    expanded_ranked = sorted(
        expanded,
        key=lambda r: float(r.get("retrieval_score") or 0.0),
        reverse=True,
    )
    parent_child_top_10 = _rows_hit_any_group(expanded_ranked[:10], expected_groups)

    pre_expansion_top_10 = rewrite_top_10_hits if query_rewrite_changed else hybrid_top_10
    parent_child_helped = bool(parent_child_top_10 and not pre_expansion_top_10)
    parent_child_hurt = bool((not parent_child_top_10) and pre_expansion_top_10)

    # Stage 4: source-tier / allowed-use filter
    deduped = _dedupe_rows(expanded_ranked)
    filtered = _apply_case_source_filter(case, deduped)
    source_filter_kept = _rows_hit_any_group(filtered, expected_groups)
    source_filter_dropped = bool(parent_child_top_10 and not source_filter_kept)

    # Stage 5: citation window (top-10 vs top-5)
    top_10 = filtered[:EVAL_CONTEXT_K]
    top_5 = filtered[:CITED_CONTEXT_K]
    cited_top_5 = _rows_hit_any_group(top_5, expected_groups)
    cited_top_10 = _rows_hit_any_group(top_10, expected_groups)

    # Oracle upper bound: fraction of distinct expected-source groups that a
    # perfect reranker could place inside top-10 from the filtered pool. This
    # uses the same fractional-recall denominator as the baseline comparison;
    # a binary case hit rate is not comparable when a case expects many sources.
    oracle_top_10 = source_filter_kept  # by definition: an expected source
    matched_filtered_groups = _matched_expected_groups(filtered, expected_groups)
    if expected_groups:
        oracle_recall_at_10 = min(
            len(matched_filtered_groups), EVAL_CONTEXT_K
        ) / len(expected_groups)
    else:
        oracle_recall_at_10 = 1.0

    final_failure_stage = _classify_final_stage(
        corpus_has_expected=corpus_has_expected,
        bm25_top_50=bm25_top_50,
        dense_top_50=dense_top_50,
        hybrid_top_50=hybrid_top_50,
        hybrid_top_10=hybrid_top_10,
        query_rewrite_hurt=query_rewrite_hurt,
        parent_child_hurt=parent_child_hurt,
        source_filter_dropped=source_filter_dropped,
        cited_top_10=cited_top_10,
        cited_top_5=cited_top_5,
        oracle_top_10=oracle_top_10,
        case=case,
    )

    return {
        "case_id": case.get("case_id"),
        "expected_intent": case.get("expected_intent"),
        "category": case.get("category"),
        "query": query,
        "rewritten_query": rewritten_query,
        "expected_source_ids": [str(s) for s in (case.get("expected_source_ids") or [])],
        "corpus_has_expected_source": corpus_has_expected,
        "bm25_top_50": bm25_top_50,
        "bm25_top_20": bm25_top_20,
        "bm25_top_10": bm25_top_10,
        "dense_top_50": dense_top_50,
        "dense_top_20": dense_top_20,
        "dense_top_10": dense_top_10,
        "hybrid_rrf_top_50": hybrid_top_50,
        "hybrid_rrf_top_20": hybrid_top_20,
        "hybrid_rrf_top_10": hybrid_top_10,
        "query_rewrite_changed_query": query_rewrite_changed,
        "query_rewrite_helped": query_rewrite_helped,
        "query_rewrite_hurt": query_rewrite_hurt,
        "parent_child_helped": parent_child_helped,
        "parent_child_hurt": parent_child_hurt,
        "source_filter_kept_expected": source_filter_kept,
        "source_filter_dropped_expected": source_filter_dropped,
        "cited_top_5_has_expected": cited_top_5,
        "cited_top_10_has_expected": cited_top_10,
        "best_possible_recall_at_10_if_oracle_reranked": round(
            oracle_recall_at_10, 4
        ),
        "final_failure_stage": final_failure_stage,
    }


def _classify_final_stage(
    *,
    corpus_has_expected: bool,
    bm25_top_50: bool,
    dense_top_50: bool,
    hybrid_top_50: bool,
    hybrid_top_10: bool,
    query_rewrite_hurt: bool,
    parent_child_hurt: bool,
    source_filter_dropped: bool,
    cited_top_10: bool,
    cited_top_5: bool,
    oracle_top_10: bool,
    case: Mapping[str, Any],
) -> str:
    if cited_top_5:
        return "no_failure"
    if not corpus_has_expected:
        return "gold_source_missing_from_corpus"
    if not (bm25_top_50 or dense_top_50 or hybrid_top_50):
        return "candidate_generation_failure"
    if not dense_top_50 and bm25_top_50:
        return "dense_failure"
    if not bm25_top_50 and dense_top_50:
        return "sparse_failure"
    if (bm25_top_50 or dense_top_50) and not hybrid_top_10 and hybrid_top_50:
        return "rrf_ranking_failure"
    if query_rewrite_hurt:
        return "query_rewrite_drift"
    if parent_child_hurt:
        return "parent_child_expansion_noise"
    if source_filter_dropped:
        return "source_filter_drop"
    if cited_top_10 and not cited_top_5:
        return "citation_window_drop"
    if oracle_top_10 and _looks_like_alias_mismatch(case):
        return "metadata_alias_mismatch"
    if not oracle_top_10 and not (case.get("expected_source_ids") or []):
        return "goldset_design_possible_issue"
    return "goldset_design_possible_issue"


# ─── Aggregation ────────────────────────────────────────────────────────


def _category_failure_breakdown(
    cases: list[dict[str, Any]],
    *,
    category_filter: str | None = None,
) -> dict[str, int]:
    out: Counter[str] = Counter()
    for c in cases:
        if c.get("final_failure_stage") == "no_failure":
            continue
        if category_filter is not None and c.get("category") != category_filter:
            continue
        out[c.get("final_failure_stage") or "unknown"] += 1
    return dict(out)


def _safe_mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def build_report(
    *,
    goldset_path: Path = DEFAULT_GOLDSET_PATH,
    actual_full_stack_recall_at_10: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    cases = _load_goldset(goldset_path)
    corpus = _knowledge_snippets()
    fingerprint = knowledge_base_fingerprint()

    per_case: list[dict[str, Any]] = []
    for case in cases:
        per_case.append(_diagnose_case(case, corpus, fingerprint))

    total = len(per_case)

    def _rate(field: str) -> float:
        if not total:
            return 0.0
        return round(sum(1 for c in per_case if c.get(field)) / total, 4)

    corpus_coverage_rate = _rate("corpus_has_expected_source")
    bm25_candidate_recall_at_50 = _rate("bm25_top_50")
    dense_candidate_recall_at_50 = _rate("dense_top_50")
    hybrid_candidate_recall_at_50 = _rate("hybrid_rrf_top_50")
    source_filter_retention_rate = _rate("source_filter_kept_expected")
    citation_window_retention_rate = _rate("cited_top_10_has_expected")
    oracle_upper_bound = _safe_mean([c["best_possible_recall_at_10_if_oracle_reranked"] for c in per_case])

    failure_stage_counts = Counter(c["final_failure_stage"] for c in per_case)
    category_failure_counts = _category_failure_breakdown(per_case)
    intent_failure_counts = Counter(
        c.get("expected_intent") or "unknown"
        for c in per_case if c.get("final_failure_stage") != "no_failure"
    )

    summary = {
        "total_n": total,
        "corpus_coverage_rate": corpus_coverage_rate,
        "bm25_candidate_recall_at_50": bm25_candidate_recall_at_50,
        "dense_candidate_recall_at_50": dense_candidate_recall_at_50,
        "hybrid_candidate_recall_at_50": hybrid_candidate_recall_at_50,
        "source_filter_retention_rate": source_filter_retention_rate,
        "citation_window_retention_rate": citation_window_retention_rate,
        "oracle_recall_at_10_upper_bound": oracle_upper_bound,
        "actual_full_stack_recall_at_10": actual_full_stack_recall_at_10,
        "oracle_gap": (
            round(oracle_upper_bound - actual_full_stack_recall_at_10, 4)
            if actual_full_stack_recall_at_10 is not None
            else None
        ),
        "failure_stage_counts": dict(failure_stage_counts),
        "category_failure_counts": category_failure_counts,
        "intent_failure_counts": dict(intent_failure_counts),
        "taglish_failure_counts": _category_failure_breakdown(per_case, category_filter="taglish"),
        "genetics_vus_failure_counts": _category_failure_breakdown(per_case, category_filter="genetics_vus"),
        "tumor_marker_failure_counts": _category_failure_breakdown(per_case, category_filter="tumor_marker"),
        "supplement_failure_counts": _category_failure_breakdown(per_case, category_filter="supplement"),
        "urgent_symptom_failure_counts": _category_failure_breakdown(per_case, category_filter="urgent_symptom"),
        "source_tier_filtering_failure_counts": _category_failure_breakdown(
            per_case, category_filter="source_tier_filtering"
        ),
    }

    return {
        "schema_version": "rag_stage_oracle_diagnostic_v1",
        "status": "informational",
        "label": "rag_stage_oracle_diagnostic",
        "clinical_validation": False,
        "claim_boundary": (
            "Engineering diagnostic only.  This artifact does not improve retrieval "
            "by itself; it attributes the stage at which expected source grounding "
            "is lost.  Live-agent generation is NOT exercised.  No clinical "
            "validation, real-world safety, or production healthcare readiness is "
            "established by this diagnostic."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "goldset_path": str(goldset_path).replace("\\", "/"),
        "wall_time_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "summary": summary,
        "cases": per_case,
        "contamination_note": (
            "Diagnostic runs against the frozen internal goldset.  Promoting any of "
            "its attribution categories into a retrieval change requires re-running "
            "the baseline comparison and confirming the trade is honest; the "
            "diagnostic itself does NOT change retrieval ranking, source governance, "
            "or any live-agent behaviour."
        ),
        "stage_vocabulary": list(FAILURE_STAGES),
    }


def _load_goldset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _baseline_full_stack_recall_at_10() -> float | None:
    """Best-effort: read the latest baseline-comparison artifact for the full stack."""
    p = Path("Data/evals/rag/latest_rag_baseline_comparison.json")
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return (data.get("summary") or {}).get("full_stack_recall_at_10")


def write_report(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    goldset_path: Path = DEFAULT_GOLDSET_PATH,
) -> Path:
    report = build_report(
        goldset_path=goldset_path,
        actual_full_stack_recall_at_10=_baseline_full_stack_recall_at_10(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


__all__ = [
    "CANDIDATE_LIMIT",
    "DEFAULT_GOLDSET_PATH",
    "DEFAULT_OUTPUT_PATH",
    "FAILURE_STAGES",
    "build_report",
    "write_report",
]
