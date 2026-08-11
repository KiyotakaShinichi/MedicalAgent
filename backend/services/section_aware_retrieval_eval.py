"""Section-aware retrieval ablation over existing internal research cases.

This module never edits the frozen retrieval goldsets.  It evaluates a
query-aware section reranker against the previously recorded section misses
and the KB-derived research-paper case set.  The result is internal tuning
evidence, not an independent holdout.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Iterable, Mapping

from backend.services.agent_rag import _knowledge_snippets, knowledge_base_fingerprint
from backend.services.rag_baseline_comparison import (
    _apply_case_source_filter,
    _bm25_only_retrieval,
    _dedupe_rows,
    _retrieve_for_config,
)
from backend.services.rag_vector_index import DEFAULT_RAG_INDEX_PATH, rag_index_status
from backend.services.research_paper_kb_eval import CASES_PATH, FULL_STACK_ID
from backend.services.section_aware_retrieval import canonical_section, rerank_by_section


ROOT = Path(__file__).resolve().parents[2]
MIXED_BANK = ROOT / "Data/evals/agentic_tool_use/mixed_query_scale_bank.jsonl"
MIXED_FAILURES = ROOT / "Data/evals/agentic_tool_use/latest_mixed_query_scale_failures.json"
DEFAULT_OUTPUT = ROOT / "Data/evals/rag/latest_section_aware_retrieval_eval.json"
DEFAULT_FAILURE_OUTPUT = ROOT / "Data/evals/rag/latest_section_aware_retrieval_failures.json"
CLAIM_BOUNDARY = (
    "Internal, KB-derived section-ranking evidence only. It is tuning-used and does not establish "
    "independent retrieval generalization, medical correctness, clinical validation, patient benefit, "
    "or production healthcare readiness."
)


def run_section_aware_retrieval_eval(
    *,
    output_path: Path | str = DEFAULT_OUTPUT,
    failure_output_path: Path | str = DEFAULT_FAILURE_OUTPUT,
) -> dict[str, Any]:
    corpus = _knowledge_snippets()
    fingerprint = knowledge_base_fingerprint()
    known_miss_cases = _known_section_miss_cases()
    canonical_cases = _load_jsonl(Path(CASES_PATH))
    search_cache: dict[tuple[str, str], tuple[list[dict[str, Any]], float]] = {}
    rewrite_cache: dict[tuple[str, str], tuple[str, float]] = {}

    known_rows = [
        _evaluate_case(
            case,
            config_id=FULL_STACK_ID,
            section_aware=True,
            corpus=corpus,
            fingerprint=fingerprint,
            search_cache=search_cache,
            rewrite_cache=rewrite_cache,
        )
        for case in known_miss_cases
    ]

    ablations: dict[str, dict[str, Any]] = {}
    for config_id, section_aware in (
        ("bm25_only", False),
        ("bm25_only", True),
        ("faiss_dense_only", False),
        ("faiss_dense_only", True),
        ("hybrid_rrf", False),
        ("hybrid_rrf", True),
        (FULL_STACK_ID, False),
        (FULL_STACK_ID, True),
    ):
        label = f"section_aware_{config_id}" if section_aware else config_id
        rows = [
            _evaluate_case(
                case,
                config_id=config_id,
                section_aware=section_aware,
                corpus=corpus,
                fingerprint=fingerprint,
                search_cache=search_cache,
                rewrite_cache=rewrite_cache,
            )
            for case in canonical_cases
        ]
        ablations[label] = {"summary": _summarize(rows), "cases": rows}

    baseline = ablations[FULL_STACK_ID]["summary"]
    candidate = ablations[f"section_aware_{FULL_STACK_ID}"]["summary"]
    section_delta = round(candidate["section_hit_rate"] - baseline["section_hit_rate"], 4)
    paper_delta = round(candidate["paper_recall_at_10"] - baseline["paper_recall_at_10"], 4)
    precision_delta = round(candidate["expected_paper_precision_at_5"] - baseline["expected_paper_precision_at_5"], 4)
    recovered = sum(row["candidate_section_hit_at_10"] and not row["baseline_section_hit_at_10"] for row in known_rows)
    regressions = sum(row["baseline_section_hit_at_10"] and not row["candidate_section_hit_at_10"] for row in known_rows)
    promoted = bool(section_delta > 0 and paper_delta >= 0 and precision_delta >= -0.02 and regressions == 0)
    index_path = Path(DEFAULT_RAG_INDEX_PATH)
    payload = {
        "schema_version": "section_aware_retrieval_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "acceptable_internal_experiment" if promoted else "needs_attention",
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "independent_holdout": False,
        "was_used_for_tuning": True,
        "known_section_miss_source": _relative(MIXED_FAILURES),
        "known_section_miss_count": len(known_rows),
        "known_miss_evaluation": {
            "recovered_misses": recovered,
            "remaining_misses": len(known_rows) - recovered,
            "regression_cases": regressions,
            "recovery_rate": round(recovered / max(len(known_rows), 1), 4),
            "cases": known_rows,
        },
        "ablation_case_file": _relative(Path(CASES_PATH)),
        "ablation_case_file_sha256": _normalized_sha256(Path(CASES_PATH)),
        "ablations": ablations,
        "decision": {
            "promoted_to_live_retrieval": promoted,
            "section_hit_rate_delta": section_delta,
            "paper_recall_at_10_delta": paper_delta,
            "expected_paper_precision_at_5_delta": precision_delta,
            "reason": (
                "Section-aware ranking improved the internal section target without paper-recall or material precision regression."
                if promoted else
                "Section-aware ranking did not clear the predeclared quality-preservation conditions and remains experimental."
            ),
        },
        "index": {
            "status": rag_index_status(corpus=corpus, knowledge_fingerprint=fingerprint),
            "serialized_size_bytes": index_path.stat().st_size if index_path.exists() else None,
            "indexing_latency_ms": None,
            "indexing_latency_note": "Existing serialized index reused; no rebuild was hidden inside this retrieval comparison.",
        },
        "limitations": [
            "The 319-style misses were already inspected and are tuning-used.",
            "The canonical ablation cases were derived from the evaluated KB.",
            "Section labels are metadata targets, not medical-correctness labels.",
            "A positive internal result must be repeated on an independently authored no-read holdout before a generalization claim.",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    failures = [row for row in known_rows if not row["candidate_section_hit_at_10"]]
    failure_payload = {
        "schema_version": "section_aware_retrieval_failures_v1",
        "generated_at": payload["generated_at"],
        "status": "needs_attention" if failures else "acceptable_internal_experiment",
        "failure_count": len(failures),
        "failures": failures,
        "clinical_validation": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(Path(output_path), payload)
    _write_json(Path(failure_output_path), failure_payload)
    return payload


def _evaluate_case(
    case: Mapping[str, Any],
    *,
    config_id: str,
    section_aware: bool,
    corpus: list[dict[str, Any]],
    fingerprint: str,
    search_cache: dict[tuple[str, str], tuple[list[dict[str, Any]], float]],
    rewrite_cache: dict[tuple[str, str], tuple[str, float]],
) -> dict[str, Any]:
    query = str(case.get("query") or case.get("user_query") or "")
    intent = "education"
    started = perf_counter()
    if config_id == "bm25_only":
        baseline_rows = _bm25_only_retrieval(query, corpus, limit=50)
        retrieval_latency_ms = (perf_counter() - started) * 1_000
    else:
        baseline_rows, retrieval_latency_ms = _retrieve_for_config(
            config_id,
            query,
            intent,
            corpus,
            fingerprint,
            search_cache,
            rewrite_cache,
        )
    baseline_rows = _dedupe_rows(baseline_rows)
    if "source_tier" in config_id:
        baseline_rows = _apply_case_source_filter(case, baseline_rows)
    candidate_started = perf_counter()
    candidate_rows = rerank_by_section(query, baseline_rows) if section_aware else list(baseline_rows)
    section_latency_ms = (perf_counter() - candidate_started) * 1_000
    expected_pmcid = str(case.get("expected_pmcid") or "").upper()
    expected_section = canonical_section(case.get("expected_section"))
    baseline_top10 = baseline_rows[:10]
    candidate_top10 = candidate_rows[:10]
    baseline_section_hit = _section_hit(baseline_top10, expected_pmcid, expected_section)
    candidate_section_hit = _section_hit(candidate_top10, expected_pmcid, expected_section)
    first_rank = _first_paper_rank(candidate_top10, expected_pmcid)
    return {
        "case_id": case.get("case_id"),
        "query": query,
        "category": case.get("category"),
        "expected_pmcid": expected_pmcid or None,
        "expected_section": None if expected_section == "unknown" else expected_section,
        "baseline_section_hit_at_10": baseline_section_hit,
        "candidate_section_hit_at_10": candidate_section_hit,
        "paper_hit_at_5": _paper_hit(candidate_rows[:5], expected_pmcid),
        "paper_hit_at_10": _paper_hit(candidate_top10, expected_pmcid),
        "first_relevant_paper_rank": first_rank,
        "mrr": round(1.0 / first_rank, 4) if first_rank else 0.0,
        "expected_paper_precision_at_5": _paper_precision(candidate_rows[:5], expected_pmcid),
        "retrieved_sections": [canonical_section(row.get("section")) for row in candidate_top10],
        "retrieved_pmcids": [str(row.get("pmcid") or "").upper() or None for row in candidate_top10],
        "latency_ms": round(retrieval_latency_ms + section_latency_ms, 3),
        "section_rerank_latency_ms": round(section_latency_ms, 4),
    }


def _known_section_miss_cases() -> list[dict[str, Any]]:
    failures = json.loads(MIXED_FAILURES.read_text(encoding="utf-8")).get("failures") or []
    ids = {
        str(row.get("case_id"))
        for row in failures
        if "expected_section_missing_at_10" in (row.get("failure_reasons") or [])
    }
    return [row for row in _load_jsonl(MIXED_BANK) if str(row.get("case_id")) in ids]


def _summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    positive = [row for row in rows if row.get("expected_pmcid")]
    section_rows = [row for row in positive if row.get("expected_section")]
    latencies = [float(row.get("latency_ms") or 0.0) for row in rows]
    return {
        "case_count": len(rows),
        "positive_case_count": len(positive),
        "section_labeled_case_count": len(section_rows),
        "paper_recall_at_5": _mean(float(row["paper_hit_at_5"]) for row in positive),
        "paper_recall_at_10": _mean(float(row["paper_hit_at_10"]) for row in positive),
        "mrr": _mean(float(row["mrr"]) for row in positive),
        "section_hit_rate": _mean(float(row["candidate_section_hit_at_10"]) for row in section_rows),
        "expected_paper_precision_at_5": _mean(float(row["expected_paper_precision_at_5"]) for row in positive),
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
    }


def _paper_hit(rows: Iterable[Mapping[str, Any]], expected_pmcid: str) -> bool:
    if not expected_pmcid:
        return False
    return any(str(row.get("pmcid") or "").upper() == expected_pmcid for row in rows)


def _section_hit(rows: Iterable[Mapping[str, Any]], expected_pmcid: str, expected_section: str) -> bool:
    if not expected_pmcid or expected_section == "unknown":
        return False
    return any(
        str(row.get("pmcid") or "").upper() == expected_pmcid
        and canonical_section(row.get("section")) == expected_section
        for row in rows
    )


def _first_paper_rank(rows: Iterable[Mapping[str, Any]], expected_pmcid: str) -> int | None:
    for index, row in enumerate(rows, start=1):
        if str(row.get("pmcid") or "").upper() == expected_pmcid:
            return index
    return None


def _paper_precision(rows: list[Mapping[str, Any]], expected_pmcid: str) -> float:
    if not rows or not expected_pmcid:
        return 0.0
    return round(sum(str(row.get("pmcid") or "").upper() == expected_pmcid for row in rows) / len(rows), 4)


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return round(sum(rows) / len(rows), 4) if rows else 0.0


def _percentile(values: list[float], pct: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(value, 3)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


__all__ = ["run_section_aware_retrieval_eval"]
