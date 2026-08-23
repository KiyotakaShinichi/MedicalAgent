"""Shared helpers scoped to the breast-monitoring integration suite."""

import uuid
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tests.breast_monitoring.environment import (
    configure_breast_monitoring_test_environment,
)

configure_breast_monitoring_test_environment()

from backend.database import Base  # noqa: E402


class FakeSeries:
    def __init__(self, role, description, instances):
        self.candidate_role = role
        self.series_description = description
        self.instance_count = instances


def _temp_db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return Session()


def _temp_root():
    test_root = Path("Data/test_tmp")
    test_root.mkdir(parents=True, exist_ok=True)
    return test_root


def _make_temp_dir(root):
    path = Path(root) / f"unit_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ─── Failure-only RAG diagnostics ────────────────────────────────────────────
#
# Two retrieval assertions pass on every developer machine and fail only on the
# Linux CI runner, and the bare assertion text ("0 not greater than or equal to
# 1", "'unideal' not found in ...") says nothing about *where* the funnel
# collapsed. Reproduction attempts eliminated corpus size, test ordering, both
# module caches, dependency versions, and Python-version ranking differences, so
# the next evidence has to come from the runner itself.
#
# These helpers render state the pipeline *already* records — nothing new is
# computed and no production code is involved. They run only while building an
# assertion message, so they cost nothing on the passing path.
#
# Deliberately excluded so the output is safe for a public Actions log: patient
# identifiers, query and reply text, document bodies, embeddings, credentials.
# Only counts, identifiers, statuses, and scores are emitted.


def _rag_pipeline_diagnostics(result):
    """Collapse the retrieval funnel into one comparable line-per-stage dict.

    The funnel is the point: `retrieved -> reranked -> compressed` plus the two
    tier filters shows which stage returned nothing, which is exactly what the
    assertion alone cannot say.
    """
    import platform

    trace = result.get("pipeline_trace") or {}
    confidence = result.get("retrieval_confidence") or {}
    pregen = result.get("pregen_tier_filter") or {}
    tier = result.get("tier_filter") or {}
    initial = pregen.get("initial_retrieval") or pregen

    return {
        "platform": platform.system(),
        "python": platform.python_version(),
        "numpy": _module_version("numpy"),
        "intent": result.get("intent"),
        "rag_mode": result.get("rag_mode"),
        "safety_level": (result.get("safety") or {}).get("level"),
        "terminal_step": trace.get("terminal_step"),
        # The funnel, in execution order.
        "retrieved_count": trace.get("retrieved_count"),
        "reranked_count": trace.get("reranked_count"),
        "compressed_count": trace.get("compressed_count"),
        "pregen_tier_kept": initial.get("kept_count"),
        "pregen_tier_dropped": initial.get("dropped_count"),
        "tier_kept": tier.get("kept_count"),
        "tier_dropped": tier.get("dropped_count"),
        "retrieval_context_count": len(result.get("retrieval_context") or []),
        "citation_count": len(result.get("citations") or []),
        # Scores that decide whether anything survives.
        "top_score": confidence.get("top_score"),
        "top_k_evaluated": confidence.get("top_k_evaluated"),
        "high_trust_chunks": confidence.get("high_trust_chunks"),
        "answerability_status": confidence.get("answerability_status"),
        "retrieval_confidence": confidence.get("retrieval_confidence"),
        # Corpus actually loaded in this process.
        "corpus_docs": _corpus_size(),
        "ingested_chunks": _ingested_chunk_count(),
        "encoder_status": _encoder_status(),
        # Which chunks survived, so a Linux/Windows run can be diffed by id.
        "kept_chunk_ids": (tier.get("kept_chunk_ids") or [])[:5],
        "pregen_kept_chunk_ids": (initial.get("kept_chunk_ids") or [])[:5],
    }


def _module_version(name):
    try:
        import importlib

        return getattr(importlib.import_module(name), "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001 - diagnostics must never raise
        return f"unavailable:{type(exc).__name__}"


def _corpus_size():
    try:
        from backend.services.agent_kb_corpus import knowledge_snippets

        return len(knowledge_snippets())
    except Exception as exc:  # noqa: BLE001
        return f"unavailable:{type(exc).__name__}"


def _ingested_chunk_count():
    try:
        from backend.services.kb_ingestion import load_ingested_chunks

        return len(load_ingested_chunks())
    except Exception as exc:  # noqa: BLE001
        return f"unavailable:{type(exc).__name__}"


def _encoder_status():
    """Whether the semantic encoder is genuinely loadable in this process.

    The remaining unexplained surface is the semantic stage, and a runtime that
    fails closed reports success from every other check, so this is asked
    directly rather than inferred.
    """
    try:
        from backend.services.dep001b_semantic_safety import classify_dep001b_safety

        prediction = classify_dep001b_safety("What is chemotherapy in general?")
        return {
            "policy_action": getattr(prediction, "policy_action", None),
            "failure_reason": getattr(prediction, "failure_reason", None),
        }
    except Exception as exc:  # noqa: BLE001
        return f"raised:{type(exc).__name__}"


def _observed(case, key):
    """Read one `observed` field without assuming the block is a mapping.

    Defensive for the same reason as :func:`_failed_check_names`: this runs
    only while a test is already failing, so a `TypeError` here would replace
    the real failure with a less useful one.
    """
    observed = case.get("observed") if isinstance(case, dict) else None
    return observed.get(key) if isinstance(observed, dict) else None


def _failed_check_names(checks):
    """Names of the checks that failed, from either shape the report may use.

    `agent_regression_eval` emits a *list* of `{"name", "passed"}` records; an
    earlier draft of this helper assumed a mapping and raised `AttributeError`
    while building the failure message, which replaced the real assertion
    failure with its own. Both shapes are handled, and anything unrecognised
    degrades to an empty list rather than raising.
    """
    if isinstance(checks, dict):
        return sorted(name for name, ok in checks.items() if ok is False)
    if isinstance(checks, list):
        return sorted(
            str(item.get("name"))
            for item in checks
            if isinstance(item, dict) and item.get("passed") is False
        )
    return []


def _format_diagnostics(label, payload):
    """Stable key order so a Linux run and a Windows run diff cleanly."""
    import json

    return f"\n{label}:\n" + json.dumps(payload, indent=2, sort_keys=True, default=str)


def _regression_failure_diagnostics(report):
    """Per-case summary for the agent regression suite.

    `status == 'unideal'` with every guardrail metric at 1.0 means
    `pass_rate < 0.80`, so the useful evidence is *which* cases failed and what
    each one retrieved — not the aggregate the assertion already showed.
    """
    import platform

    summary = report.get("summary") or {}
    cases = report.get("cases") or report.get("results") or []
    failed = [c for c in cases if str(c.get("status")).lower() in {"failed", "fail"}]

    return {
        "platform": platform.system(),
        "python": platform.python_version(),
        "status": summary.get("status"),
        "case_count": report.get("case_count"),
        "pass_rate": summary.get("pass_rate"),
        "attack_block_rate": summary.get("attack_block_rate"),
        "output_guardrail_pass_rate": summary.get("output_guardrail_pass_rate"),
        "expected_source_hit_rate": summary.get("expected_source_hit_rate"),
        "citation_presence_rate": summary.get("citation_presence_rate"),
        "corpus_docs": _corpus_size(),
        "ingested_chunks": _ingested_chunk_count(),
        "failed_case_count": len(failed),
        "failed_cases": [
            {
                "id": c.get("id"),
                "category": c.get("category"),
                "failed_checks": _failed_check_names(c.get("checks")),
                "intent": _observed(c, "intent"),
                "retrieval_context_ids": (_observed(c, "retrieval_context_ids") or [])[:5],
                "citation_ids": (_observed(c, "citation_ids") or [])[:5],
                "grounding_score": _observed(c, "grounding_score"),
            }
            for c in failed[:8]
        ],
    }
