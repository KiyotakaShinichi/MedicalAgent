"""Internal paired evaluation for claim-conditioned citation selection."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.claim_conditioned_citation_selector import (
    select_citations_for_claims,
)


DEFAULT_OUTPUT_PATH = Path("Data/evals/rag/latest_claim_conditioned_citation_selector_eval.json")


def build_claim_conditioned_citation_selector_eval(
    output_path: str | Path | None = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    rows = [_evaluate_case(case) for case in _cases()]
    answerable = [row for row in rows if not row["refusal_route"]]
    baseline_precision = _mean(row["baseline_precision"] for row in answerable)
    selector_precision = _mean(row["selector_precision"] for row in answerable)
    baseline_support = _mean(row["baseline_support"] for row in answerable)
    selector_support = _mean(row["selector_support"] for row in answerable)
    governance_pass = all(row["disallowed_selected_count"] == 0 for row in rows)
    refusal_pass = all(
        not row["selected_ids"] for row in rows if row["refusal_route"]
    )
    paired_improvement = (
        selector_precision > baseline_precision
        and selector_support >= baseline_support
        and governance_pass
        and refusal_pass
    )
    report = {
        "schema_version": "claim_conditioned_citation_selector_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "acceptable_internal_candidate" if paired_improvement else "needs_attention",
        "internal_vs_external": "internal_authored_tuning_used",
        "was_used_for_tuning": True,
        "case_count": len(rows),
        "baseline_top3_citation_precision": round(baseline_precision, 4),
        "selector_citation_precision": round(selector_precision, 4),
        "citation_precision_delta": round(selector_precision - baseline_precision, 4),
        "baseline_claim_support_rate": round(baseline_support, 4),
        "selector_claim_support_rate": round(selector_support, 4),
        "disallowed_source_selection_count": sum(row["disallowed_selected_count"] for row in rows),
        "refusal_citation_strip_passed": refusal_pass,
        "paired_internal_improvement": paired_improvement,
        "promotion_decision": "offline_shadow_candidate_only" if paired_improvement else "do_not_promote",
        "live_patient_route_changed": False,
        "support_proxy_is_entailment": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "cases": rows,
        "claim_boundary": (
            "Internal engineering comparison only. Cases were authored with the selector and are tuning-used. "
            "This is not held-out evidence, not semantic entailment, not clinical validation, and not a reason "
            "to enable the selector on patient-facing routes without generated-answer shadow evaluation."
        ),
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    chunks = case["chunks"]
    expected = set(case["expected_source_ids"])
    baseline = [str(chunk["source_id"]) for chunk in chunks[:3]]
    result = select_citations_for_claims(
        case["claims"], chunks, refusal_route=case.get("refusal_route", False)
    )
    selected = result["selected_citation_ids"]
    baseline_precision = len(set(baseline) & expected) / len(baseline) if baseline else 0.0
    selector_precision = len(set(selected) & expected) / len(selected) if selected else (1.0 if case.get("refusal_route") else 0.0)
    return {
        "case_id": case["case_id"],
        "refusal_route": bool(case.get("refusal_route")),
        "expected_source_ids": sorted(expected),
        "baseline_ids": baseline,
        "selected_ids": selected,
        "baseline_precision": round(baseline_precision, 4),
        "selector_precision": round(selector_precision, 4),
        "baseline_support": float(bool(set(baseline) & expected)),
        "selector_support": float(bool(set(selected) & expected)) if not case.get("refusal_route") else 1.0,
        "disallowed_selected_count": sum(
            source_id in selected and chunk.get("allowed_use") == "clinician_only"
            for source_id, chunk in ((str(chunk["source_id"]), chunk) for chunk in chunks)
        ),
        "unsupported_claims": result["unsupported_claims"],
    }


def _cases() -> list[dict[str, Any]]:
    return [
        _case("her2", "HER2-positive refers to higher HER2 protein levels in cancer cells.", "her2-source", "CBC includes white blood cells.", "Medication timing should be reviewed."),
        _case("cbc", "A CBC measures blood components including white blood cells and platelets.", "cbc-source", "HER2 is a breast cancer biomarker.", "Imaging reports describe scan findings."),
        _case("vus", "A variant of uncertain significance should not be treated as a positive result.", "vus-source", "Tumor markers need clinical context.", "A CBC includes platelets."),
        _case("marker", "A tumor marker change alone does not prove recurrence.", "marker-source", "Genetic counseling reviews hereditary results.", "Symptoms can be logged for review."),
        {
            "case_id": "refusal",
            "claims": ["Change the treatment dose."],
            "expected_source_ids": [],
            "refusal_route": True,
            "chunks": [_chunk("dose-protocol", "Change treatment dose using this protocol.", 1.0, allowed_use="clinician_only")],
        },
    ]


def _case(case_id: str, claim: str, expected_id: str, distractor_a: str, distractor_b: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "claims": [claim],
        "expected_source_ids": [expected_id],
        "refusal_route": False,
        "chunks": [
            _chunk(f"{case_id}-distractor-a", distractor_a, 0.99),
            _chunk(f"{case_id}-distractor-b", distractor_b, 0.95),
            _chunk(expected_id, claim, 0.75),
            _chunk(f"{case_id}-clinician", claim, 1.0, allowed_use="clinician_only"),
        ],
    }


def _chunk(source_id: str, text: str, score: float, *, allowed_use: str = "general_patient_education") -> dict[str, Any]:
    return {
        "source_id": source_id,
        "text": text,
        "retrieval_score": score,
        "source_tier": "T1" if allowed_use != "clinician_only" else "T4",
        "allowed_use": allowed_use,
    }


def _mean(values) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


__all__ = ["build_claim_conditioned_citation_selector_eval"]
