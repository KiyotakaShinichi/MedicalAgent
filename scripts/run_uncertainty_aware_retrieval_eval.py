"""Probe the uncertainty-aware retrieval routing layer.

Replays a small hermetic probe set (no FAISS, no LLM) — each probe is
a triple of (chunks, claim_envelope, safety_envelope) plus a label for
the expected ``answerability_status``.  Writes
``Data/evals/rag/latest_uncertainty_aware_retrieval_eval.json``.

The probe set is intentionally seeded by hand so the eval is
reproducible across CI runs and machines without retrieval indices.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.retrieval_confidence import (  # noqa: E402
    ANSWERABILITY_STATUS_VALUES,
    classify_retrieval_uncertainty,
)


OUTPUT_PATH = Path("Data/evals/rag/latest_uncertainty_aware_retrieval_eval.json")


def _chunks(*scores_and_tiers: tuple[float, str]) -> list[dict[str, Any]]:
    return [{"score": s, "source_tier": t, "text": "stub"} for s, t in scores_and_tiers]


def _verdicts(supported: int = 0, contradicted: int = 0, unsupported: int = 0) -> dict[str, Any]:
    out: list[dict[str, Any]] = []
    out.extend({"status": "supported"} for _ in range(supported))
    out.extend({"status": "contradicted"} for _ in range(contradicted))
    out.extend({"status": "unsupported"} for _ in range(unsupported))
    return {"verdicts": out}


def _probes() -> list[dict[str, Any]]:
    return [
        {
            "probe_id": "URC-001",
            "label": "answerable_with_citations",
            "intent": "education",
            "chunks": _chunks((0.85, "T1"), (0.8, "T1"), (0.75, "T2")),
            "claims": _verdicts(supported=4),
            "safety": {"level": "low_risk", "scope": "education_or_tracking"},
        },
        {
            "probe_id": "URC-002",
            "label": "answerable_with_limited_context",
            "intent": "education",
            "chunks": _chunks((0.35, "T2"), (0.3, "T3"), (0.25, "T4")),
            "claims": _verdicts(supported=2, unsupported=1),
            "safety": {"level": "low_risk", "scope": "education_or_tracking"},
        },
        {
            "probe_id": "URC-003",
            "label": "insufficient_evidence",
            "intent": "education",
            "chunks": _chunks((0.05, "T5")),
            "claims": _verdicts(unsupported=3),
            "safety": {"level": "low_risk", "scope": "education_or_tracking"},
        },
        {
            "probe_id": "URC-004",
            "label": "conflicting_evidence",
            "intent": "education",
            "chunks": _chunks((0.7, "T1"), (0.6, "T2")),
            "claims": _verdicts(supported=2, contradicted=1),
            "safety": {"level": "low_risk", "scope": "education_or_tracking"},
        },
        {
            "probe_id": "URC-005",
            "label": "clinician_review_required",
            "intent": "record_explanation",
            "chunks": _chunks((0.7, "T1"), (0.6, "T2")),
            "claims": _verdicts(supported=1, unsupported=3),
            "safety": {"level": "low_risk", "scope": "education_or_tracking"},
        },
        {
            "probe_id": "URC-006",
            "label": "refuse_due_to_safety",
            "intent": "education",
            "chunks": _chunks((0.9, "T1"), (0.85, "T1")),
            "claims": _verdicts(supported=4),
            "safety": {"level": "high_risk", "scope": "treatment_decision_request"},
        },
        {
            "probe_id": "URC-007",
            "label": "refuse_due_to_safety",
            "intent": "education",
            "chunks": _chunks((0.9, "T1"), (0.85, "T1")),
            "claims": _verdicts(supported=4),
            "safety": {"level": "high_risk", "scope": "diagnosis_or_outcome_claim"},
        },
        {
            "probe_id": "URC-008",
            "label": "refuse_due_to_safety",
            "intent": "education",
            "chunks": _chunks((0.9, "T1")),
            "claims": _verdicts(supported=1),
            "safety": {"level": "high_risk", "scope": "urgent_or_safety_related"},
        },
        {
            "probe_id": "URC-009",
            "label": "answerable_with_citations",
            "intent": "education",
            "chunks": _chunks((0.9, "T1"), (0.8, "T1"), (0.7, "T1")),
            "claims": _verdicts(supported=3),
            "safety": {"level": "low_risk", "scope": "education_or_tracking"},
        },
        {
            "probe_id": "URC-010",
            "label": "insufficient_evidence",
            "intent": "education",
            "chunks": _chunks((0.5, "T4"), (0.4, "T4")),
            "claims": _verdicts(supported=1, unsupported=4),
            "safety": {"level": "low_risk", "scope": "education_or_tracking"},
        },
    ]


def run() -> dict[str, Any]:
    probes = _probes()
    results: list[dict[str, Any]] = []
    correct = 0
    for probe in probes:
        outcome = classify_retrieval_uncertainty(
            chunks=probe["chunks"],
            claim_envelope=probe["claims"],
            safety=probe["safety"],
            intent=probe.get("intent"),
        )
        passed = outcome.answerability_status == probe["label"]
        if passed:
            correct += 1
        results.append({
            "probe_id": probe["probe_id"],
            "intent": probe["intent"],
            "expected": probe["label"],
            "actual": outcome.answerability_status,
            "passed": passed,
            "envelope": outcome.to_dict(),
        })
    summary = {
        "schema_version": "1.0",
        "status": "informational",
        "label": "internal_engineering_eval_curated_probe_set",
        "claim_boundary": (
            "These probes are a curated, hand-authored set used to verify the "
            "routing logic of retrieval_confidence.py.  A pass_rate of 1.0 "
            "means the routing decisions match the labels — it does NOT "
            "establish clinical validity or real-world retrieval quality."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_probes": len(probes),
        "n_passed": correct,
        "total_n": len(probes),
        "pass_count": correct,
        "fail_count": len(probes) - correct,
        "skipped_count": 0,
        "pass_rate": correct / len(probes),
        "answerability_status_values": list(ANSWERABILITY_STATUS_VALUES),
        "probes": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    summary = run()
    print(f"wrote: {OUTPUT_PATH}")
    print(f"  n={summary['n_probes']}  passed={summary['n_passed']}  rate={summary['pass_rate']:.3f}")
    return 0 if summary["pass_rate"] == 1.0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
