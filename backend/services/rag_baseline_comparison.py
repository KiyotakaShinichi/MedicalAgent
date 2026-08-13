"""Frozen-goldset RAG baseline comparison.

This module evaluates retrieval/grounding configurations against the existing
retrieval goldset without changing the live agent path.  The goal is to show
whether the more complex RAG stack earns its complexity against simpler
baselines, while keeping the result framed as engineering evidence only.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

from backend.services.agent_query_rewriting import rewrite_and_decompose
from backend.services.agent_rag import _knowledge_snippets, knowledge_base_fingerprint
from backend.services.agent_retrieval import expand_parent_child_windows
from backend.services.kb_source_governance import ALLOWED_USE_VOCABULARY
from backend.services.rag_intent_modes import COMMON_BANNED_CLAIMS, RagModeConfig
from backend.services.rag_tier_filter import filter_chunks_by_mode
from backend.services.rag_vector_index import rag_index_status, search_hybrid_index


ROOT_DIR = Path(__file__).resolve().parents[2]
GOLDSET_PATH = ROOT_DIR / "Data/evals/rag/retrieval_goldset.jsonl"
COMPARISON_OUTPUT_PATH = ROOT_DIR / "Data/evals/rag/latest_rag_baseline_comparison.json"
FAILURES_OUTPUT_PATH = ROOT_DIR / "Data/evals/rag/latest_rag_baseline_failures.json"

CITED_CONTEXT_K = 5
EVAL_CONTEXT_K = 10
INITIAL_CANDIDATE_LIMIT = 50
PARENT_CHILD_SEED_LIMIT = 20

CONFIGURATIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "bm25_only",
        "label": "BM25 only",
        "description": "Sparse BM25 lexical retrieval over the frozen KB corpus; no query rewriting.",
    },
    {
        "id": "faiss_dense_only",
        "label": "FAISS dense only",
        "description": "Dense/vector score ordering from the active local index; falls back honestly if dense FAISS is unavailable.",
    },
    {
        "id": "hybrid_rrf",
        "label": "Dense + sparse hybrid RRF",
        "description": "Active dense/sparse hybrid retrieval with reciprocal-rank-style fusion; no query rewriting.",
    },
    {
        "id": "hybrid_rrf_query_rewrite",
        "label": "Hybrid + query rewrite",
        "description": "Hybrid retrieval using the agent query rewriting/decomposition output.",
    },
    {
        "id": "hybrid_rrf_query_rewrite_parent_child",
        "label": "Hybrid + rewrite + parent-child",
        "description": "Hybrid retrieval with query rewriting plus parent-child context expansion.",
    },
    {
        "id": "hybrid_rrf_query_rewrite_parent_child_source_tier",
        "label": "Hybrid + rewrite + parent-child + source tiers",
        "description": "Full compared retrieval stack with source-tier/allowed-use filtering before context selection.",
    },
    {
        "id": "hybrid_rrf_query_rewrite_parent_child_source_tier_pruned",
        "label": "Full stack + citation-context pruner",
        "experimental": True,
        "positioning": "negative_result_not_promoted",
        "description": (
            "Full stack with the citation_context_pruner applied between source-tier "
            "filtering and citation assembly.  Eval-path experiment only — not wired "
            "into the live patient agent."
        ),
    },
)

REFUSAL_INTENTS = {
    "urgent_escalation",
    "genetic_counselor_review",
    "tumor_marker_boundary",
    "pharmacist_or_clinician_review",
    "treatment_refusal",
    "prognosis_refusal",
    "diagnosis_refusal",
    "privacy_refusal",
}

# Source normalization keeps logical gold labels comparable to the current KB
# source IDs.  It is not a ranking tweak and does not inspect retrieved text.
LOGICAL_SOURCE_ALIASES: dict[str, set[str]] = {
    "nci-her2-breast": {
        "nci-her2-breast",
        "her2 in breast cancer",
        "breast-treatment-basics",
        "national cancer institute",
    },
    "curated-her2-basics": {
        "curated-her2-basics",
        "nci-her2-breast",
        "her2 in breast cancer",
        "breast-treatment-basics",
    },
    "cbc-monitoring": {
        "cbc-monitoring",
        "curated-wbc-neutropenia",
        "cbc labs and trend monitoring",
        "cbc, anc, hemoglobin, and platelet monitoring reference",
        "0185db088c803c80",
        "36b7a3ffdb9205a4",
        "927cf11805df9019d710",
        "f6726c194bf1f479171f",
    },
    "curated-wbc-neutropenia": {
        "curated-wbc-neutropenia",
        "cbc-monitoring",
        "side effects and red flags during breast cancer treatment",
        "treatment-side-effects",
        "3ca1dfefbd3147b0",
        "c30ab0b49f328562e76f",
        "nci-febrile-neutropenia",
        "febrile neutropenia during chemotherapy",
    },
    "infection-safety": {
        "infection-safety",
        "cdc",
        "cdc-fever-chemo",
        "fever during chemotherapy",
        "nci-febrile-neutropenia",
        "febrile neutropenia during chemotherapy",
        "treatment-side-effects",
        # Discovered by content match (see latest_source_alias_coverage.json).
        "9a6347c207d53299",  # Hematology, Bleeding, and Infection Review Reference
    },
    "curated-fever-neutropenia": {
        "curated-fever-neutropenia",
        "nci-febrile-neutropenia",
        "febrile neutropenia during chemotherapy",
        "infection-safety",
        "treatment-side-effects",
    },
    "imaging-monitoring": {
        "imaging-monitoring",
        "curated-mri-response-terms",
        "imaging report monitoring: mri, ct, ultrasound, and response language",
        "mri, ct, ultrasound, and imaging response terms reference",
        "a734a844daed9ef7",
        "33ef73acba84d60bd7a1",
        "87ec22bc66c88b40ea76",
        "7cd1e3e1103a156a",
    },
    "curated-mri-response-terms": {
        "curated-mri-response-terms",
        "imaging-monitoring",
        "imaging report monitoring: mri, ct, ultrasound, and response language",
        "mri, ct, ultrasound, and imaging response terms reference",
        "a734a844daed9ef7",
        "33ef73acba84d60bd7a1",
        "87ec22bc66c88b40ea76",
        "7cd1e3e1103a156a",
        # Discovered by content match (see latest_source_alias_coverage.json).
        "2524619e8115a75d",  # DCE-MRI texture features for early breast cancer therapy response prediction
        "2a9f2ed73f0b189c",  # Early treatment response prediction using DCE-MRI tumor heterogeneity
    },
    "genetic-counseling": {
        "genetic-counseling",
        "curated-vus-boundary",
        "genetic counseling readiness and family history intake",
        "germline testing, somatic testing, vus, and multigene panels",
        "genetics, biomarker, and tumor marker safety terms reference",
        "22d463a5a12d490af4c6",
        "29f0f5dda9789b7e",
        "4787d2a42440789f",
        "eafe5c100c4cd819b6fa",
        "917264d81e3123c0d2a8",
        # Discovered by content match against KB titles
        # (see latest_source_alias_coverage.json, 2026-05-27 diagnostic).
        "664fb49bb1343408",  # Family History Readiness Depth Reference
        "ef3bcc511aad3c2c",  # Genetic Counseling Readiness and Family History Intake
    },
    "curated-vus-boundary": {
        "curated-vus-boundary",
        "genetic-counseling",
        "vus",
        "germline testing, somatic testing, vus, and multigene panels",
        "genetics, biomarker, and tumor marker safety terms reference",
        "29f0f5dda9789b7e",
        "4787d2a42440789f",
        "eafe5c100c4cd819b6fa",
        "917264d81e3123c0d2a8",
    },
    "tumor-marker-context": {
        "tumor-marker-context",
        "curated-tumor-marker-limitations",
        "minimum evidence and medical claim boundaries",
        "genetics, biomarker, and tumor marker safety terms reference",
        "28cfcee61ce1e4a4",
        "4787d2a42440789f",
        "972b1b8be879098562a7",
        "150bf2854b59cec640b1",
        "917264d81e3123c0d2a8",
        # Discovered by content match (see latest_source_alias_coverage.json).
        "5598e2371d2713c4",  # Breast Cancer Biomarkers and Tumor Marker Safety
    },
    "curated-tumor-marker-limitations": {
        "curated-tumor-marker-limitations",
        "tumor-marker-context",
        "minimum evidence and medical claim boundaries",
        "genetics, biomarker, and tumor marker safety terms reference",
        "28cfcee61ce1e4a4",
        "4787d2a42440789f",
        "972b1b8be879098562a7",
        "150bf2854b59cec640b1",
        "917264d81e3123c0d2a8",
        # Discovered by content match (see latest_source_alias_coverage.json).
        "5598e2371d2713c4",  # Breast Cancer Biomarkers and Tumor Marker Safety
    },
    "supplement-safety": {
        "supplement-safety",
        "curated-st-johns-wort",
        "curated-st-johns-wort-safety",
        "nci-msk-supplement-safety",
        "supplements during cancer treatment",
        "curated supplement interaction safety",
        "supplement and natural product safety by product reference",
        "6649c1bba1cd7799",
        "2c9cf580eb45af0e",
        "bd077c510af8e9bb2107",
        # Discovered by content match (see latest_source_alias_coverage.json).
        "918edc260afd2d63",  # Diagnosis, Treatment, and Supplement Safety Boundaries
    },
    "curated-st-johns-wort": {
        "curated-st-johns-wort",
        "curated-st-johns-wort-safety",
        "supplement-safety",
        "st. johns wort interaction safety",
        "st johns wort interaction safety",
    },
    "project safety policy": {
        "project safety policy",
        "project-monitoring-score",
        "monitoring score boundary",
        "diagnosis, treatment, and supplement safety boundaries",
        "minimum evidence and medical claim boundaries",
        "response-modeling",
        "918edc260afd2d63",
        "28cfcee61ce1e4a4",
        "b4b9ee5dfff5d9bb4a84",
    },
    "treatment-side-effects": {
        "treatment-side-effects",
        "acs-chemo-side-effects",
        "side effects and red flags during breast cancer treatment",
        "3ca1dfefbd3147b0",
        "1d8b472e73bcd9696d15",
        # Discovered by content match (see latest_source_alias_coverage.json).
        "24de6c8ad0379f43",  # GI Symptoms, Mouth Sores, Neuropathy, and Fatigue Reference
        "d50090fd5d38a39d",  # Symptom Red Flags and Review Hints During Treatment
    },
    "portal-help": {
        "portal-help",
        "portal-help-upload",
        "portal-help-symptom-entry",
        "portal-help-lab-results",
        "portal-help-mri-upload",
        "patient portal help",
        "using the patient portal tools",
        # Discovered by content match (see latest_source_alias_coverage.json).
        "c35c9264029ff9c9",  # NLCare Portal Help and Data Entry
        "479e2ce02e7d9e05",  # Patient Portal Workflow Reference
    },
}


def run_rag_baseline_comparison(
    *,
    goldset_path: Path | str = GOLDSET_PATH,
    comparison_output_path: Path | str = COMPARISON_OUTPUT_PATH,
    failures_output_path: Path | str = FAILURES_OUTPUT_PATH,
) -> dict[str, Any]:
    """Run all retrieval baselines and persist comparison + failure artifacts."""

    goldset = _load_goldset(Path(goldset_path))
    corpus = _knowledge_snippets()
    fingerprint = knowledge_base_fingerprint()
    index_status_before = rag_index_status(corpus=corpus, knowledge_fingerprint=fingerprint)

    configurations: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []

    search_cache: dict[tuple[str, str], tuple[list[dict[str, Any]], float]] = {}
    rewrite_cache: dict[tuple[str, str], tuple[str, float]] = {}

    for config in CONFIGURATIONS:
        result = _evaluate_configuration(config, goldset, corpus, fingerprint, search_cache, rewrite_cache)
        configurations[config["id"]] = result
        failures = [case for case in result["cases"] if case["failure_reasons"]]
        all_failures.extend(
            {
                "configuration": config["id"],
                "configuration_label": config["label"],
                **_failure_projection(case),
            }
            for case in failures
        )
        summary = result["summary"]
        comparison_rows.append({
            "configuration": config["id"],
            "label": config["label"],
            "experimental": bool(config.get("experimental", False)),
            "positioning": config.get("positioning", "canonical_comparison"),
            "recall_at_5": summary["recall_at_5"],
            "recall_at_10": summary["recall_at_10"],
            "mrr": summary["mrr"],
            "ndcg_at_10": summary["ndcg_at_10"],
            "citation_precision": summary["citation_precision"],
            "claim_support_rate": summary["claim_support_rate"],
            "unsupported_context_rate": summary["unsupported_context_rate"],
            "refusal_correctness": summary["refusal_correctness"],
            "source_tier_correctness": summary["source_tier_correctness"],
            "latency_p50_ms": summary["latency_p50_ms"],
            "latency_p95_ms": summary["latency_p95_ms"],
            "failure_count": len(failures),
            "failure_examples": [_failure_projection(case) for case in failures[:3]],
        })

    simple = configurations["bm25_only"]["summary"]
    full = configurations["hybrid_rrf_query_rewrite_parent_child_source_tier"]["summary"]
    best = max(comparison_rows, key=lambda row: (row["recall_at_10"], row["mrr"], -row["unsupported_context_rate"]))
    improvement_over_bm25 = round(full["recall_at_10"] - simple["recall_at_10"], 4)
    status = "acceptable" if full["source_tier_correctness"] >= 1.0 and full["unsupported_context_rate"] <= 0.25 else "needs_attention"

    payload = {
        "schema_version": "rag_baseline_comparison_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "goldset_path": _repo_relative(Path(goldset_path)),
        "total_n": len(goldset),
        "authored_by": "engineering",
        "authored_date": _goldset_authored_date(goldset),
        "was_used_for_tuning": False,
        "internal_vs_external_authored": "internal",
        "clinical_validation": False,
        "baseline_version": "retrieval_goldset_v1_2026_05",
        "contamination_note": (
            "This suite uses the frozen internal RAG retrieval goldset. It compares retrieval "
            "configurations only; do not tune on this file or present the result as external validation."
        ),
        "source_id_normalization_note": (
            "Logical source aliases normalize gold labels to current KB source IDs. They do not change ranking."
        ),
        "index_status": rag_index_status(corpus=corpus, knowledge_fingerprint=fingerprint),
        "index_status_before_eval": index_status_before,
        "configurations": configurations,
        "rows": comparison_rows,
        "summary": {
            "case_count": len(goldset),
            "best_configuration": best["configuration"],
            "best_recall_at_10": best["recall_at_10"],
            "bm25_recall_at_10": simple["recall_at_10"],
            "full_stack_recall_at_10": full["recall_at_10"],
            "complex_stack_improvement_over_bm25": improvement_over_bm25,
            "full_stack_mrr": full["mrr"],
            "full_stack_ndcg_at_10": full["ndcg_at_10"],
            "citation_precision": full["citation_precision"],
            "claim_support_rate": full["claim_support_rate"],
            "unsupported_context_rate": full["unsupported_context_rate"],
            "refusal_correctness": full["refusal_correctness"],
            "source_tier_correctness": full["source_tier_correctness"],
            "latency_p50_ms": full["latency_p50_ms"],
            "latency_p95_ms": full["latency_p95_ms"],
            "improvement_proven_vs_bm25": improvement_over_bm25 > 0 and full["unsupported_context_rate"] <= simple["unsupported_context_rate"],
            "engineering_evidence_only": True,
        },
        "claim_boundary": (
            "RAG baseline comparison is engineering evidence that the retrieval stack finds and filters "
            "sources better than simpler baselines on an internal frozen goldset. It is not clinical "
            "validation, medical accuracy proof, or real-world safety evidence."
        ),
    }

    failure_payload = _build_failure_payload(goldset, all_failures)
    _write_json(Path(comparison_output_path), payload)
    _write_json(Path(failures_output_path), failure_payload)
    return payload


def _evaluate_configuration(
    config: Mapping[str, Any],
    cases: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    fingerprint: str,
    search_cache: dict[tuple[str, str], tuple[list[dict[str, Any]], float]],
    rewrite_cache: dict[tuple[str, str], tuple[str, float]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for case in cases:
        query = str(case.get("user_query") or case.get("query") or "")
        intent = _map_goldset_intent(str(case.get("expected_intent") or "education"))
        ranked, latency_ms = _retrieve_for_config(config["id"], query, intent, corpus, fingerprint, search_cache, rewrite_cache)
        ranked = _dedupe_rows(ranked)
        if "source_tier" in config["id"]:
            filter_started = time.perf_counter()
            ranked = _apply_case_source_filter(case, ranked)
            latency_ms += (time.perf_counter() - filter_started) * 1000.0
        if config["id"].endswith("_pruned"):
            # Apply the citation-context pruner AFTER source-governance
            # filtering and BEFORE the top-k window used by citation
            # precision.  We prune to EVAL_CONTEXT_K so the
            # downstream recall / MRR / NDCG metrics are computed on
            # the same window size as the unpruned baseline — any
            # recall regression is therefore attributable to the
            # pruner's ordering, not to a smaller window.
            from backend.services.citation_context_pruner import prune as _prune_citation_context
            pruner_started = time.perf_counter()
            rewritten_query = _rewritten_query_for(query, intent, rewrite_cache)
            ranked = _prune_citation_context(
                ranked,
                query=query,
                rewritten_query=rewritten_query,
                intent=str(case.get("expected_intent") or intent or ""),
                keep=EVAL_CONTEXT_K,
                refusal_route=bool(case.get("expected_refusal_or_insufficient_evidence")),
            )
            latency_ms += (time.perf_counter() - pruner_started) * 1000.0
        top10 = ranked[:EVAL_CONTEXT_K]
        latencies.append(latency_ms)
        rows.append(_score_case(config["id"], case, top10, latency_ms))

    summary = {
        "case_count": len(rows),
        "recall_at_5": _mean(row["recall_at_5"] for row in rows),
        "recall_at_10": _mean(row["recall_at_10"] for row in rows),
        "mrr": _mean(row["mrr"] for row in rows),
        "ndcg_at_10": _mean(row["ndcg_at_10"] for row in rows),
        "citation_precision": _mean(row["citation_precision"] for row in rows),
        "claim_support_rate": _rate(row["claim_supported"] for row in rows),
        "unsupported_context_rate": _rate(row["unsupported_context"] for row in rows),
        "refusal_correctness": _rate(row["refusal_correct"] for row in rows),
        "source_tier_correctness": _rate(row["source_tier_correct"] for row in rows),
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
    }
    return {"summary": summary, "cases": rows}


def _rewritten_query_for(
    query: str,
    intent: str,
    rewrite_cache: dict[tuple[str, str], tuple[str, float]],
) -> str:
    """Reuse a cached rewritten query if available; otherwise return ``query``.

    The baseline-comparison flow rewrites once per (query, intent) pair
    and caches the result.  The pruner wants the rewritten form to
    compute query-overlap; pulling from the same cache keeps latency
    flat and avoids rerunning the LLM rewriter.
    """
    cached = rewrite_cache.get((query, intent))
    if cached:
        return cached[0]
    return query


def _retrieve_for_config(
    config_id: str,
    query: str,
    intent: str,
    corpus: list[dict[str, Any]],
    fingerprint: str,
    search_cache: dict[tuple[str, str], tuple[list[dict[str, Any]], float]],
    rewrite_cache: dict[tuple[str, str], tuple[str, float]],
) -> tuple[list[dict[str, Any]], float]:
    if config_id == "bm25_only":
        started = time.perf_counter()
        return _bm25_only_retrieval(query, corpus, limit=INITIAL_CANDIDATE_LIMIT), (time.perf_counter() - started) * 1000.0

    search_query = query
    rewrite_latency_ms = 0.0
    if "query_rewrite" in config_id:
        rewrite_key = (query, intent)
        if rewrite_key not in rewrite_cache:
            rewrite_started = time.perf_counter()
            rewritten = rewrite_and_decompose(query, intent)
            rewrite_cache[rewrite_key] = (
                str(rewritten.get("expanded_query") or query),
                (time.perf_counter() - rewrite_started) * 1000.0,
            )
        search_query, rewrite_latency_ms = rewrite_cache[rewrite_key]

    search_key = (search_query, intent)
    if search_key not in search_cache:
        search_started = time.perf_counter()
        search_cache[search_key] = (search_hybrid_index(
            query=search_query,
            corpus=corpus,
            intent=intent,
            knowledge_fingerprint=fingerprint,
            candidate_limit=INITIAL_CANDIDATE_LIMIT,
        ), (time.perf_counter() - search_started) * 1000.0)
    cached_candidates, search_latency_ms = search_cache[search_key]
    candidates = list(cached_candidates)

    if config_id == "faiss_dense_only":
        ranked = sorted(
            candidates,
            key=lambda row: float(row.get("dense_score") if row.get("dense_score") is not None else row.get("vector_score") or 0.0),
            reverse=True,
        )
    else:
        ranked = sorted(candidates, key=lambda row: float(row.get("retrieval_score") or 0.0), reverse=True)

    if "parent_child" in config_id:
        expansion_started = time.perf_counter()
        seed = ranked[:PARENT_CHILD_SEED_LIMIT]
        expanded = expand_parent_child_windows(seed)
        expansion_latency_ms = (time.perf_counter() - expansion_started) * 1000.0
        return (
            sorted(expanded, key=lambda row: float(row.get("retrieval_score") or 0.0), reverse=True),
            rewrite_latency_ms + search_latency_ms + expansion_latency_ms,
        )
    return ranked, rewrite_latency_ms + search_latency_ms


def _score_case(
    config_id: str,
    case: Mapping[str, Any],
    ranked: list[dict[str, Any]],
    latency_ms: float,
) -> dict[str, Any]:
    expected_groups = _expected_source_groups(case)
    raw_expected = [str(value) for value in case.get("expected_source_ids", []) if str(value).strip()]
    matched5 = _matched_expected_groups(ranked[:5], expected_groups)
    matched10 = _matched_expected_groups(ranked[:10], expected_groups)
    first_rank = _first_relevant_rank(ranked, expected_groups)
    expected_refusal = bool(case.get("expected_refusal_or_insufficient_evidence"))
    refusal_correct = _refusal_policy_correct(case)
    citation_precision = _citation_precision(ranked[:CITED_CONTEXT_K], expected_groups, expected_refusal)
    source_tier_correct = _source_tier_correct(case, ranked[:EVAL_CONTEXT_K])
    recall_at_5 = len(matched5) / max(len(expected_groups), 1)
    recall_at_10 = len(matched10) / max(len(expected_groups), 1)
    unsupported_context = recall_at_10 == 0.0
    claim_supported = recall_at_10 > 0.0
    failure_reasons = _failure_reasons(
        recall_at_10=recall_at_10,
        citation_precision=citation_precision,
        source_tier_correct=source_tier_correct,
        refusal_correct=refusal_correct,
    )

    return {
        "case_id": case.get("case_id"),
        "configuration": config_id,
        "query": case.get("user_query") or case.get("query"),
        "expected_intent": case.get("expected_intent"),
        "expected_refusal_or_insufficient_evidence": expected_refusal,
        "expected_source_ids": raw_expected,
        "retrieved_source_ids": [_representative_row_id(row) for row in ranked[:EVAL_CONTEXT_K]],
        "first_relevant_rank": first_rank,
        "recall_at_5": round(recall_at_5, 4),
        "recall_at_10": round(recall_at_10, 4),
        "mrr": round(1.0 / first_rank, 4) if first_rank else 0.0,
        "ndcg_at_10": _ndcg_at_10(ranked, expected_groups),
        "citation_precision": citation_precision,
        "claim_supported": claim_supported,
        "unsupported_context": unsupported_context,
        "refusal_correct": refusal_correct,
        "refusal_correctness_note": "Policy/intent proxy; generated-answer refusal is covered by live-agent safety evals.",
        "source_tier_correct": source_tier_correct,
        "latency_ms": round(latency_ms, 3),
        "failure_reasons": failure_reasons,
    }


def _bm25_only_retrieval(query: str, corpus: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    tokens = [_tokenize(_document_text(row)) for row in corpus]
    query_tokens = _tokenize(query)
    try:
        from rank_bm25 import BM25Okapi

        scores = list(BM25Okapi(tokens).get_scores(query_tokens))
    except Exception:
        query_terms = set(query_tokens)
        scores = [len(query_terms & set(row_tokens)) / max(1, len(query_terms)) for row_tokens in tokens]

    rows: list[dict[str, Any]] = []
    for item, score in zip(corpus, scores):
        if float(score) <= 0:
            continue
        rows.append({
            **item,
            "retrieval_score": round(float(score), 4),
            "retrieval_backend": "bm25_only",
            "backend": "bm25_only",
            "sparse_score": round(float(score), 4),
        })
    return sorted(rows, key=lambda row: float(row.get("retrieval_score") or 0.0), reverse=True)[:limit]


def _apply_case_source_filter(case: Mapping[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mode = _case_filter_mode(case)
    result = filter_chunks_by_mode(rows, mode, keep_unmapped=False)
    return result.kept_chunks


def _case_filter_mode(case: Mapping[str, Any]) -> RagModeConfig:
    allowed_tiers = tuple(str(tier) for tier in (case.get("acceptable_source_tiers") or ["T1", "T2", "T3"]))
    allowed_use = tuple(use for use in ALLOWED_USE_VOCABULARY if use != "clinician_only")
    return RagModeConfig(
        mode="baseline_comparison_patient_facing_filter",
        description="Eval-only patient-facing source-tier and allowed-use filter.",
        audience="patient",
        allowed_tiers=allowed_tiers,
        allowed_use=allowed_use,
        allow_citations=True,
        insufficient_evidence_default="Insufficient sourced evidence; route to care team review.",
        banned_claim_categories=COMMON_BANNED_CLAIMS,
        max_retrieved_chunks=EVAL_CONTEXT_K,
        require_clinician_handoff_clause=bool(case.get("expected_refusal_or_insufficient_evidence")),
    )


def _source_tier_correct(case: Mapping[str, Any], rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return True
    result = filter_chunks_by_mode(rows, _case_filter_mode(case), keep_unmapped=False)
    return len(result.kept_chunks) == len(rows)


def _expected_source_groups(case: Mapping[str, Any]) -> list[set[str]]:
    raw_ids = [str(value).strip().lower() for value in case.get("expected_source_ids", []) if str(value).strip()]
    groups: list[set[str]] = []
    for value in raw_ids:
        aliases = {value}
        aliases |= LOGICAL_SOURCE_ALIASES.get(value, set())
        aliases |= LOGICAL_SOURCE_ALIASES.get(value.lower(), set())
        groups.append({_normalize_identifier(item) for item in aliases if item})
    return groups


def _matched_expected_groups(rows: list[dict[str, Any]], expected_groups: list[set[str]]) -> set[int]:
    matched: set[int] = set()
    for row in rows:
        ids = _row_ids(row)
        for index, group in enumerate(expected_groups):
            if ids & group:
                matched.add(index)
    return matched


def _first_relevant_rank(rows: list[dict[str, Any]], expected_groups: list[set[str]]) -> int | None:
    for rank, row in enumerate(rows[:EVAL_CONTEXT_K], start=1):
        if any(_row_ids(row) & group for group in expected_groups):
            return rank
    return None


def _citation_precision(rows: list[dict[str, Any]], expected_groups: list[set[str]], expected_refusal: bool) -> float:
    if not rows:
        return 1.0 if expected_refusal else 0.0
    relevant = sum(1 for row in rows if any(_row_ids(row) & group for group in expected_groups))
    return round(relevant / max(len(rows), 1), 4)


def _ndcg_at_10(rows: list[dict[str, Any]], expected_groups: list[set[str]]) -> float:
    if not expected_groups:
        return 0.0
    seen_groups: set[int] = set()
    dcg = 0.0
    for rank, row in enumerate(rows[:EVAL_CONTEXT_K], start=1):
        ids = _row_ids(row)
        newly_matched = [index for index, group in enumerate(expected_groups) if index not in seen_groups and ids & group]
        if newly_matched:
            seen_groups.add(newly_matched[0])
            dcg += 1.0 / math.log2(rank + 1)
    ideal_count = min(len(expected_groups), EVAL_CONTEXT_K)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return round(dcg / ideal, 4) if ideal else 0.0


def _row_ids(row: Mapping[str, Any]) -> set[str]:
    values = {
        row.get("id"),
        row.get("chunk_id"),
        row.get("source_id"),
        row.get("source_name"),
        row.get("parent_id"),
        row.get("title"),
        row.get("source_url"),
        row.get("topic"),
    }
    ids = {_normalize_identifier(value) for value in values if value}
    expanded = set(ids)
    for value in ids:
        expanded |= {_normalize_identifier(item) for item in LOGICAL_SOURCE_ALIASES.get(value, set())}
    return expanded


def _representative_row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("source_id") or row.get("id") or row.get("parent_id") or row.get("source_name") or "")


def _failure_reasons(
    *,
    recall_at_10: float,
    citation_precision: float,
    source_tier_correct: bool,
    refusal_correct: bool,
) -> list[str]:
    reasons: list[str] = []
    if recall_at_10 <= 0:
        reasons.append("retrieval_miss")
        reasons.append("unsupported_context")
    elif recall_at_10 < 1.0:
        reasons.append("partial_source_recall")
    if citation_precision < 0.5:
        reasons.append("low_citation_precision")
    if not source_tier_correct:
        reasons.append("source_tier_mismatch")
    if not refusal_correct:
        reasons.append("refusal_policy_mismatch")
    return reasons


def _failure_projection(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "query": case.get("query"),
        "expected_intent": case.get("expected_intent"),
        "expected_source_ids": case.get("expected_source_ids"),
        "retrieved_source_ids": case.get("retrieved_source_ids"),
        "first_relevant_rank": case.get("first_relevant_rank"),
        "recall_at_10": case.get("recall_at_10"),
        "citation_precision": case.get("citation_precision"),
        "failure_reasons": case.get("failure_reasons"),
    }


def _build_failure_payload(goldset: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    by_configuration = Counter(str(item.get("configuration")) for item in failures)
    by_reason: Counter[str] = Counter()
    for item in failures:
        by_reason.update(str(reason) for reason in item.get("failure_reasons") or [])
    return {
        "schema_version": "rag_baseline_failures_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "needs_attention" if failures else "strong",
        "total_n": len(goldset),
        "failed_n": len(failures),
        "by_configuration": dict(sorted(by_configuration.items())),
        "by_reason": dict(sorted(by_reason.items())),
        "failures": failures,
        "clinical_validation": False,
        "claim_boundary": (
            "Failure examples are for engineering triage only. They do not establish clinical behavior."
        ),
    }


def _load_goldset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"RAG retrieval goldset is missing: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"RAG retrieval goldset is empty: {path}")
    return rows


def _goldset_authored_date(rows: list[Mapping[str, Any]]) -> str | None:
    dates = sorted({str(row.get("authored_date")) for row in rows if row.get("authored_date")})
    return dates[-1] if dates else None


def _map_goldset_intent(intent: str) -> str:
    if intent in {"urgent_escalation"}:
        return "safety_boundary"
    if intent in {
        "treatment_refusal",
        "prognosis_refusal",
        "diagnosis_refusal",
        "genetic_counselor_review",
        "tumor_marker_boundary",
        "pharmacist_or_clinician_review",
    }:
        return "safety_boundary"
    if intent == "privacy_refusal":
        return "security_boundary"
    return intent or "education"


def _refusal_policy_correct(case: Mapping[str, Any]) -> bool:
    expected_refusal = bool(case.get("expected_refusal_or_insufficient_evidence"))
    intent = str(case.get("expected_intent") or "")
    actual_refusal = intent in REFUSAL_INTENTS
    return actual_refusal == expected_refusal


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("id") or row.get("chunk_id") or row.get("source_id") or row.get("parent_id"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _document_text(item: Mapping[str, Any]) -> str:
    return " ".join([
        str(item.get("title") or ""),
        str(item.get("text") or ""),
        " ".join(str(tag) for tag in item.get("tags") or []),
        str(item.get("topic") or ""),
        str(item.get("section") or ""),
    ])


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9][a-zA-Z0-9/-]+", (text or "").lower())


def _normalize_identifier(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    return round(sum(vals) / max(len(vals), 1), 4)


def _rate(values: Iterable[bool]) -> float:
    vals = [bool(value) for value in values]
    return round(sum(1 for value in vals if value) / max(len(vals), 1), 4)


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if percentile == 50:
        return round(float(median(ordered)), 3)
    index = math.ceil((percentile / 100) * len(ordered)) - 1
    return round(ordered[max(0, min(index, len(ordered) - 1))], 3)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


__all__ = [
    "COMPARISON_OUTPUT_PATH",
    "FAILURES_OUTPUT_PATH",
    "GOLDSET_PATH",
    "run_rag_baseline_comparison",
]
