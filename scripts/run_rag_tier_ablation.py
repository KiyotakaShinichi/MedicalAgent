"""Run the Phase 11 RAG source-tier ablation against the **live** agent.

Sweeps four tier configurations — T1 only / T1+T2 / T1+T2+T3 / all —
over the canonical 28-case RAG eval set (``evals/rag_eval_cases.json``)
and reports per-config metrics (pass_rate, claim_support_rate,
citation_precision, source_tier_correctness, refusal_correctness,
unsafe_answer_rate, latency_p50_ms).

Each tier config monkey-patches ``rag_intent_modes.MODES`` to restrict
``allowed_tiers`` on the relevant modes (education_rag /
record_explanation_rag / clinician_context_rag) for the duration of the
run, then invokes ``run_patient_agent_pipeline`` against an in-memory
SQLite session.

Older history
~~~~~~~~~~~~~
This script previously used a deterministic stub agent that mirrored
expected behavior so the harness could exercise the scoring contract
end-to-end without the live RAG stack.  The stub is preserved as
``--stub`` for fast smoke-tests; the default is now the live agent so
the artifact reflects real retrieval + claim-validation behavior.

Usage
~~~~~
    python scripts/run_rag_tier_ablation.py             # live agent (slow, real)
    python scripts/run_rag_tier_ablation.py --stub      # legacy deterministic stub (fast)
    python scripts/run_rag_tier_ablation.py --limit 6   # first 6 cases only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force the sparse retrieval backend by default — keeps the script
# runnable without a dense embedding model download.  Users who want
# the dense path can unset this before invoking.
os.environ.setdefault("RAG_FORCE_SPARSE", "true")

from backend.services.rag_intent_aware_eval import EVAL_CASES, load_canonical_cases
from backend.services.rag_tier_ablation import run_tier_ablation


# ─── Stub agent (legacy --stub mode) ────────────────────────────────────────


def _stub_factory_for(cases):
    """Deterministic stub: returns a hand-rolled envelope mirroring the
    expected per-case behavior.  Useful for fast contract testing of the
    ablation harness without exercising retrieval."""
    by_query = {case["query"]: case for case in cases}

    def _factory(allowed_tiers: tuple[str, ...]):
        def _call(query: str) -> dict:
            case = by_query.get(query, {})
            intent = case.get("expected_intent") or "education"
            rag_mode = case.get("expected_mode") or "education_rag"
            expects_refusal = bool(case.get("expects_refusal")) or intent in {
                "safety_boundary",
                "treatment_decision_boundary",
            }
            source_tier = allowed_tiers[0] if allowed_tiers else "T4"
            reply = (
                "This is monitoring support only - not a diagnosis or treatment recommendation. "
                "Please discuss this with your oncology care team."
                if expects_refusal
                else "Here is patient-safe education with a source-backed citation for review."
            )
            grade = "insufficient" if expects_refusal else "high"
            return {
                "intent": intent,
                "rag_mode": rag_mode,
                "mode_allowed_tiers": list(allowed_tiers),
                "reply": reply,
                "evidence_grade": {
                    "grade": grade,
                    "claim_support_rate": 1.0,
                    "citation_status": "citations_supported",
                    "source_basis": [{"source_id": f"tier-{source_tier.lower()}-demo", "tier": source_tier}],
                },
                "post_gen_validator": {"decision": "allowed"},
            }
        return _call

    return _factory


# ─── Live agent factory ─────────────────────────────────────────────────────


@contextmanager
def _restrict_modes_to_tiers(allowed_tiers: tuple[str, ...]) -> Iterator[None]:
    """Temporarily replace ``allowed_tiers`` on the retrieval modes so the
    live agent's mode → tier-filter → claim-validation path runs against
    the restricted set.  Restores on exit even if the agent raises."""
    from backend.services import rag_intent_modes

    target_modes = {"education_rag", "record_explanation_rag", "clinician_context_rag"}
    original = dict(rag_intent_modes.MODES)
    try:
        replacements = {}
        for name, cfg in original.items():
            if name in target_modes:
                # Intersect the mode's original allowed_tiers with the sweep config.
                restricted = tuple(t for t in cfg.allowed_tiers if t in allowed_tiers)
                if not restricted:
                    restricted = (allowed_tiers[0],) if allowed_tiers else cfg.allowed_tiers
                replacements[name] = replace(cfg, allowed_tiers=restricted)
            else:
                replacements[name] = cfg
        rag_intent_modes.MODES.clear()
        rag_intent_modes.MODES.update(replacements)
        yield
    finally:
        rag_intent_modes.MODES.clear()
        rag_intent_modes.MODES.update(original)


def _live_factory():
    """Returns a factory that, given an allowed-tier tuple, returns an
    agent callable invoking ``run_patient_agent_pipeline`` against a
    fresh in-memory SQLite session and an in-process mode override."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from backend.models import Base
    from backend.services.agent_rag import run_patient_agent_pipeline

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)

    def _factory(allowed_tiers: tuple[str, ...]):
        def _call(query: str) -> dict:
            db = SessionLocal()
            try:
                with _restrict_modes_to_tiers(allowed_tiers):
                    result = run_patient_agent_pipeline(
                        db=db,
                        patient_id=f"TIER-ABLATION-{'_'.join(allowed_tiers) or 'none'}",
                        query=query,
                        patient_context={},
                        fallback_response="I can explain general terms.",
                    )
                # Project to the eval's expected envelope shape.
                return {
                    "intent":             result.get("intent"),
                    "rag_mode":           result.get("rag_mode"),
                    "mode_allowed_tiers": result.get("mode_allowed_tiers") or list(allowed_tiers),
                    "reply":              result.get("reply"),
                    "evidence_grade":     result.get("evidence_grade") or {},
                    "post_gen_validator": result.get("post_gen_validator") or {},
                    "refusal_type":       result.get("refusal_type"),
                }
            finally:
                db.close()
        return _call

    return _factory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG source-tier ablation runner")
    parser.add_argument("--stub", action="store_true", help="Use the deterministic stub agent (fast).")
    parser.add_argument("--limit", type=int, default=0, help="Cap on case count (0 = all).")
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not overwrite the artifact (Data/evals/rag/latest_rag_tier_ablation.json). "
             "Use for --limit smoke tests so the release gate keeps reading the full-case run.",
    )
    args = parser.parse_args(argv)

    canonical = load_canonical_cases()
    cases = canonical if canonical else EVAL_CASES
    source_label = "canonical_28" if canonical else "phase11_demo"
    if args.limit:
        cases = cases[: args.limit]

    factory = _stub_factory_for(cases) if args.stub else _live_factory()
    agent_label = "stub" if args.stub else "live"
    print(f"[tier-ablation] agent={agent_label} cases={len(cases)} ({source_label})", file=sys.stderr)

    # Smoke runs write to a tmp path so the release gate's view of the
    # full-case artifact is preserved.
    import tempfile
    output_path = (
        Path(tempfile.gettempdir()) / "oncotrack_tier_ablation_smoke.json"
        if args.no_write
        else None
    )
    if output_path is not None:
        payload = run_tier_ablation(agent_factory=factory, cases=cases, output_path=str(output_path))
    else:
        payload = run_tier_ablation(agent_factory=factory, cases=cases)
    print(json.dumps({
        "status": payload.get("status"),
        "agent": agent_label,
        "case_source": source_label,
        "case_count": len(cases),
        "tier_configs_evaluated": payload.get("tier_configs_evaluated"),
        "per_config": [
            {
                "config": c["config"],
                "pass_rate": c.get("pass_rate"),
                "claim_support_rate": c.get("claim_support_rate"),
                "citation_precision": c.get("citation_precision"),
                "source_tier_correctness": c.get("source_tier_correctness"),
                "refusal_correctness": c.get("refusal_correctness"),
                "unsafe_answer_rate": c.get("unsafe_answer_rate"),
                "post_gen_validator_trigger_rate": c.get("post_gen_validator_trigger_rate"),
                "latency_p50_ms": c.get("latency_p50_ms"),
            }
            for c in payload.get("per_config", [])
        ],
    }, indent=2))
    return 0 if payload.get("status") in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
