"""Failure projection, input loading, and persistence for RAG baselines."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from backend.services.rag_baseline_config import REFUSAL_INTENTS


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


def _build_failure_payload(
    goldset: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
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
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
