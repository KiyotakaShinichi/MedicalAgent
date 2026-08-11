"""Frozen answer-level evaluation for claim-conditioned citation selection.

The source answers predate the citation selector.  A fixture is created once
from those existing answers plus a governed retrieval snapshot, then protected
by a SHA-256 digest.  This remains an internal offline comparison: it is not an
external holdout, semantic entailment, or evidence for clinical use.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from backend.services.agent_rag import _knowledge_snippets, knowledge_base_fingerprint
from backend.services.claim_conditioned_citation_selector import (
    DISALLOWED_USES,
    select_citations_for_claims,
)
from backend.services.rag_baseline_comparison import (
    EVAL_CONTEXT_K,
    _apply_case_source_filter,
    _dedupe_rows,
    _expected_source_groups,
    _map_goldset_intent,
    _representative_row_id,
    _retrieve_for_config,
    _row_ids,
)
from backend.services.structured_claim_shadow_eval import split_atomic_claims


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ANSWERS_PATH = ROOT / "Data/evals/rag/latest_rag_gold_eval.json"
DEFAULT_FIXTURE_PATH = (
    ROOT / "Data/evals/rag/claim_conditioned_citation_selector_holdout_v1.jsonl"
)
DEFAULT_OUTPUT_PATH = (
    ROOT / "Data/evals/rag/latest_claim_conditioned_citation_selector_holdout.json"
)
FULL_STACK_ID = "hybrid_rrf_query_rewrite_parent_child_source_tier"
MIN_CASES = 30
BOOTSTRAP_SAMPLES = 2000

CLAIM_BOUNDARY = (
    "Internal answer-level engineering comparison over pre-existing offline "
    "generated answers and a frozen governed retrieval snapshot. It is not an "
    "external-author evaluation, semantic entailment, clinical validation, or "
    "permission to enable the selector on patient-facing routes."
)


def freeze_selector_holdout(
    output_path: str | Path = DEFAULT_FIXTURE_PATH,
    *,
    source_answers_path: str | Path = SOURCE_ANSWERS_PATH,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create the fixture once; refuse accidental rewrites by default."""

    target = Path(output_path)
    if target.exists() and not overwrite:
        return {
            "status": "already_frozen",
            "fixture_path": _relative(target),
            "case_count": len(_read_jsonl(target)),
            "fixture_sha256": _sha256(target),
            "overwritten": False,
        }

    source_path = Path(source_answers_path)
    source = _read_json(source_path)
    source_cases = [row for row in source.get("cases") or [] if isinstance(row, dict)]
    corpus = _knowledge_snippets()
    fingerprint = knowledge_base_fingerprint()
    search_cache: dict[tuple[str, str], tuple[list[dict[str, Any]], float]] = {}
    rewrite_cache: dict[tuple[str, str], tuple[str, float]] = {}
    frozen_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []

    for index, source_case in enumerate(source_cases, start=1):
        query = str(source_case.get("input") or "").strip()
        answer = str(source_case.get("reply") or "").strip()
        if not query or not answer:
            continue
        expected_sources = [
            str(value)
            for value in source_case.get("expected_sources") or []
            if str(value).strip()
        ]
        refusal_route = bool(source_case.get("requires_refusal"))
        intent = _map_goldset_intent(str(source_case.get("intent") or "education"))
        ranked, latency_ms = _retrieve_for_config(
            FULL_STACK_ID,
            query,
            intent,
            corpus,
            fingerprint,
            search_cache,
            rewrite_cache,
        )
        filter_case = {
            "expected_intent": source_case.get("intent") or "education",
            "expected_refusal_or_insufficient_evidence": refusal_route,
            "expected_source_ids": expected_sources,
            "acceptable_source_tiers": ["T1", "T2", "T3"],
        }
        ranked = _apply_case_source_filter(filter_case, _dedupe_rows(ranked))
        chunks = [_selector_chunk(row) for row in ranked[:EVAL_CONTEXT_K]]
        claims = split_atomic_claims(answer)
        rows.append(
            {
                "case_id": f"selector_holdout_{index:03d}",
                "source_case_id": source_case.get("case_id"),
                "query": query,
                "answer_text": answer,
                "atomic_claims": claims,
                "expected_source_ids": expected_sources,
                "refusal_route": refusal_route,
                "retrieved_chunks": chunks,
                "retrieval_latency_ms": round(latency_ms, 3),
                "retrieval_configuration": FULL_STACK_ID,
                "answer_source_artifact": _relative(source_path),
                "answer_source_generated_at": source.get("generated_at"),
                "answer_predates_selector": True,
                "internal_vs_external": "internal_frozen_preexisting_outputs",
                "was_used_for_selector_tuning": False,
                "upstream_cases_may_have_informed_other_rag_tuning": True,
                "frozen_at": frozen_at,
                "knowledge_base_fingerprint": fingerprint,
                "clinical_validation": False,
            }
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "status": "frozen",
        "fixture_path": _relative(target),
        "case_count": len(rows),
        "fixture_sha256": _sha256(target),
        "overwritten": bool(overwrite),
    }


def build_selector_holdout_eval(
    fixture_path: str | Path = DEFAULT_FIXTURE_PATH,
    output_path: str | Path | None = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    fixture = Path(fixture_path)
    rows = _read_jsonl(fixture)
    cases = [_evaluate_case(row) for row in rows]
    answerable = [row for row in cases if not row["refusal_route"]]
    baseline_precision = _mean(row["baseline_precision"] for row in answerable)
    selector_precision = _mean(row["selector_precision"] for row in answerable)
    baseline_support = _mean(row["baseline_support"] for row in answerable)
    selector_support = _mean(row["selector_support"] for row in answerable)
    baseline_unsupported = _mean(row["baseline_unsupported"] for row in answerable)
    selector_unsupported = _mean(row["selector_unsupported"] for row in answerable)
    deltas = [row["selector_precision"] - row["baseline_precision"] for row in answerable]
    ci_low, ci_high = _bootstrap_mean_ci(deltas)
    refusal_pass = all(not row["selected_ids"] for row in cases if row["refusal_route"])
    governance_pass = all(row["disallowed_selected_count"] == 0 for row in cases)
    immutable = bool(rows) and all(
        row.get("was_used_for_selector_tuning") is False for row in rows
    )
    noninferior = (
        len(cases) >= MIN_CASES
        and immutable
        and selector_precision >= baseline_precision
        and selector_support >= baseline_support
        and selector_unsupported <= baseline_unsupported
        and ci_low >= -0.02
        and refusal_pass
        and governance_pass
    )
    strict_improvement = noninferior and ci_low > 0.0
    report = {
        "schema_version": "claim_conditioned_citation_selector_holdout_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "eligible_for_live_shadow_only"
            if strict_improvement
            else "noninferior_internal_holdout"
            if noninferior
            else "needs_attention"
        ),
        "promotion_decision": (
            "live_shadow_only"
            if strict_improvement
            else "offline_only_not_promoted"
        ),
        "fixture_path": _relative(fixture),
        "fixture_sha256": _sha256(fixture) if fixture.exists() else None,
        "case_count": len(cases),
        "minimum_case_count": MIN_CASES,
        "internal_vs_external": "internal_frozen_preexisting_outputs",
        "was_used_for_selector_tuning": False,
        "upstream_cases_may_have_informed_other_rag_tuning": True,
        "baseline_top3_citation_precision": round(baseline_precision, 4),
        "selector_citation_precision": round(selector_precision, 4),
        "citation_precision_delta": round(selector_precision - baseline_precision, 4),
        "citation_precision_delta_bootstrap_95_ci": {
            "low": round(ci_low, 4),
            "high": round(ci_high, 4),
            "samples": BOOTSTRAP_SAMPLES,
            "seed": 20260811,
        },
        "baseline_claim_support_rate": round(baseline_support, 4),
        "selector_claim_support_rate": round(selector_support, 4),
        "baseline_unsupported_context_rate": round(baseline_unsupported, 4),
        "selector_unsupported_context_rate": round(selector_unsupported, 4),
        "disallowed_source_selection_count": sum(
            row["disallowed_selected_count"] for row in cases
        ),
        "refusal_citation_strip_passed": refusal_pass,
        "governance_passed": governance_pass,
        "noninferiority_contract_passed": noninferior,
        "strict_improvement_proven": strict_improvement,
        "live_patient_route_changed": False,
        "support_proxy_is_entailment": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "cases": cases,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _evaluate_case(case: Mapping[str, Any]) -> dict[str, Any]:
    chunks = [dict(row) for row in case.get("retrieved_chunks") or []]
    claims = [str(value) for value in case.get("atomic_claims") or [] if str(value)]
    refusal = bool(case.get("refusal_route"))
    expected_case = {"expected_source_ids": case.get("expected_source_ids") or []}
    groups = _expected_source_groups(expected_case)
    baseline_chunks = chunks[:3]
    selected = select_citations_for_claims(claims, chunks, refusal_route=refusal)
    selected_ids = selected.get("selected_citation_ids") or []
    selected_chunks = _chunks_for_ids(chunks, selected_ids)
    baseline_precision = _precision(baseline_chunks, groups, refusal)
    selector_precision = _precision(selected_chunks, groups, refusal)
    baseline_support = float(refusal or _has_relevant_source(baseline_chunks, groups))
    selector_support = float(refusal or _has_relevant_source(selected_chunks, groups))
    selector_all_claims_supported = float(
        refusal or _assignments_supported(selected, chunks, groups)
    )
    disallowed = sum(
        1
        for row in selected_chunks
        if str(row.get("allowed_use") or "").lower() in DISALLOWED_USES
        or bool(row.get("stale") or row.get("is_stale"))
    )
    return {
        "case_id": case.get("case_id"),
        "source_case_id": case.get("source_case_id"),
        "refusal_route": refusal,
        "claim_count": len(claims),
        "expected_source_ids": case.get("expected_source_ids") or [],
        "baseline_ids": [_representative_row_id(row) for row in baseline_chunks],
        "selected_ids": selected_ids,
        "baseline_precision": round(baseline_precision, 4),
        "selector_precision": round(selector_precision, 4),
        "baseline_support": baseline_support,
        "selector_support": selector_support,
        "selector_all_claims_supported": selector_all_claims_supported,
        "baseline_unsupported": float(not refusal and baseline_support == 0.0),
        "selector_unsupported": float(not refusal and selector_support == 0.0),
        "disallowed_selected_count": disallowed,
        "unsupported_claim_count": len(selected.get("unsupported_claims") or []),
    }


def _selector_chunk(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "id",
            "chunk_id",
            "source_id",
            "parent_id",
            "title",
            "topic",
            "source_name",
            "text",
            "content",
            "snippet",
            "retrieval_score",
            "source_tier",
            "allowed_use",
            "stale",
            "is_stale",
        )
        if row.get(key) is not None
    }


def _chunks_for_ids(
    chunks: list[dict[str, Any]], selected_ids: Iterable[str]
) -> list[dict[str, Any]]:
    remaining = [str(value) for value in selected_ids]
    output: list[dict[str, Any]] = []
    for source_id in remaining:
        match = next(
            (
                row
                for row in chunks
                if source_id
                in {
                    str(row.get("parent_id") or ""),
                    str(row.get("source_id") or ""),
                    str(row.get("id") or ""),
                    str(row.get("chunk_id") or ""),
                }
            ),
            None,
        )
        if match is not None:
            output.append(match)
    return output


def _precision(
    chunks: list[dict[str, Any]], groups: list[set[str]], refusal: bool
) -> float:
    if not chunks:
        return 1.0 if refusal else 0.0
    relevant = sum(1 for row in chunks if any(_row_ids(row) & group for group in groups))
    return relevant / len(chunks)


def _has_relevant_source(
    chunks: list[dict[str, Any]], groups: list[set[str]]
) -> bool:
    return bool(chunks) and any(
        any(_row_ids(row) & group for group in groups) for row in chunks
    )


def _assignments_supported(
    result: Mapping[str, Any], chunks: list[dict[str, Any]], groups: list[set[str]]
) -> bool:
    assignments = result.get("claim_assignments") or []
    if not assignments:
        return False
    for assignment in assignments:
        selected = [row.get("source_id") for row in assignment.get("selected_sources") or []]
        selected_chunks = _chunks_for_ids(chunks, selected)
        if not selected_chunks or not any(
            any(_row_ids(row) & group for group in groups) for row in selected_chunks
        ):
            return False
    return True


def _bootstrap_mean_ci(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(20260811)
    means = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]


def _mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return sum(rows) / len(rows) if rows else 0.0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


__all__ = ["build_selector_holdout_eval", "freeze_selector_holdout"]
