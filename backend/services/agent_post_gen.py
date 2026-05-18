"""Post-generation steps in the patient agent pipeline.

After ``generate_answer`` produces the candidate reply, two more layers
must run before the response leaves the agent:

  1. ``apply_post_gen_validator`` — the LAST safety net.  Pattern-matches
     the candidate reply for diagnosis / treatment / prognosis / dosage /
     genetic / tumor-marker overclaims.  When blocked, swaps in a
     deterministic refusal AND strips citations (a blocked output can't
     legitimately cite anything).

  2. ``apply_intent_aware_rag_layer`` — resolves the RAG mode for the
     classified intent, tier-filters the retrieved chunks against the
     mode's governance contract, runs the claim-level citation
     validator, grades the evidence, and on insufficient-evidence
     substitutes the mode's default reply.

Both functions mutate ``result`` in place; the orchestrator
(:func:`agent_rag._finalize_result`) calls them in sequence.

Public contract preserved
~~~~~~~~~~~~~~~~~~~~~~~~~
The underscore aliases ``_apply_post_gen_validator`` /
``_apply_intent_aware_rag_layer`` are kept so the ``agent_rag``
re-import shim and the existing single-call-site path in
``_finalize_result`` continue to work without rewrites.
"""
from __future__ import annotations

from typing import Any, Mapping


# ─── Layer 1: post-gen safety validator ──────────────────────────────────────


def apply_post_gen_validator(
    result: dict[str, Any],
    output_guardrails: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any] | None, Any]:
    """Run the post-generation validator on ``result["reply"]`` and apply
    the decision in place.

    Returns ``(maybe-updated output_guardrails, pgv_decision)``.  The
    ``output_guardrails`` dict is *replaced* (not mutated) when the
    validator blocks, so the caller holds the right reference.

    The post-gen check is the **LAST** chance to refuse a diagnosis /
    treatment / prognosis / dosage / genetic / tumor-marker overclaim
    that slipped through generation.  Keeping it as a named function
    makes the failure surface explicit in the call site.
    """
    from backend.services.post_generation_validator import validate_reply

    pgv_decision = validate_reply(result.get("reply") or "")
    if pgv_decision.decision == "blocked":
        original_reply = result.get("reply")
        result["reply"] = pgv_decision.suggested_response
        # A blocked output cannot legitimately cite anything; strip
        # citations so a downstream reader doesn't see "here is the
        # source for our refusal."
        result["citations"] = []
        result["post_gen_validator"] = {
            "decision": "blocked",
            "triggered_rules": pgv_decision.triggered_rules,
            "matched_excerpts": pgv_decision.matched_excerpts,
            "medical_claim_boundary": pgv_decision.claim_boundary,
            "original_reply_preview": (original_reply or "")[:240],
        }
        # Surface the block in the output-guardrail block too so existing
        # consumers (RAG eval, trace log) see it without a new field.
        if isinstance(output_guardrails, dict):
            output_guardrails = dict(output_guardrails)
            output_guardrails["status"] = "blocked_by_post_gen_validator"
            existing_issues = list(output_guardrails.get("issues") or [])
            existing_issues.extend(
                f"post_gen::{rule}" for rule in pgv_decision.triggered_rules
            )
            output_guardrails["issues"] = existing_issues
    else:
        result["post_gen_validator"] = {
            "decision": "allowed",
            "triggered_rules": [],
            "medical_claim_boundary": pgv_decision.claim_boundary,
        }
    return output_guardrails, pgv_decision


# ─── Layer 2: intent-aware RAG envelope ──────────────────────────────────────


def apply_intent_aware_rag_layer(
    result: dict[str, Any],
    retrieved: list[Any],
    input_guardrails: Mapping[str, Any] | None,
    pgv_decision: Any,
) -> None:
    """Resolve the RAG mode, filter retrieved chunks against the mode's
    tier + allowed_use rules, run the claim-level citation validator,
    grade the evidence, and on insufficient-evidence substitute the
    mode's default reply.

    Mutates ``result`` in place with: ``rag_mode``, ``mode_allowed_tiers``,
    ``mode_allowed_use``, ``tier_filter``, ``claim_validation``,
    ``evidence_grade``, and (on insufficient-evidence)
    ``insufficient_evidence_substitution``.

    Side-effect rule: when ``grade == "insufficient"`` AND the post-gen
    validator did **not** already block, substitute the mode's
    insufficient_evidence_default reply and strip citations.  This is
    the Phase 11 "insufficient evidence is a first-class outcome"
    promise.

    Wrapped in try/except: this layer must never crash chat — on any
    exception we set ``evidence_grade.grade = "missing"`` with a reason
    and return.
    """
    try:
        from backend.services.rag_claim_validator import validate_claims
        from backend.services.rag_evidence_grading import grade_evidence
        from backend.services.rag_intent_modes import select_mode
        from backend.services.rag_tier_filter import filter_chunks_by_mode

        actor_role = (
            result.get("actor_role")
            or (input_guardrails or {}).get("actor_role")
        )
        mode = select_mode(result.get("intent"), actor_role=actor_role)
        if mode is None:
            return

        chunks_for_filter = result.get("retrieval_context") or retrieved or []
        filter_result = filter_chunks_by_mode(chunks_for_filter, mode)
        claim_validation = validate_claims(
            result.get("reply") or "",
            filter_result.kept_chunks,
        )
        grade = grade_evidence(
            mode=mode,
            filter_result=filter_result,
            claim_validation=claim_validation,
            retrieved_count_before_filter=len(chunks_for_filter),
        )
        result["rag_mode"] = mode.mode
        result["mode_allowed_tiers"] = list(mode.allowed_tiers)
        result["mode_allowed_use"] = list(mode.allowed_use)
        result["tier_filter"] = filter_result.to_dict()
        result["claim_validation"] = claim_validation.to_dict()
        result["evidence_grade"] = grade.to_dict()

        # Substitute the mode's insufficient-evidence default when
        # grading collapses AND the validator hadn't already blocked.
        if (
            grade.grade == "insufficient"
            and pgv_decision.decision != "blocked"
            and mode.insufficient_evidence_default
        ):
            result["reply"] = mode.insufficient_evidence_default
            result["citations"] = []
            result["insufficient_evidence_substitution"] = {
                "reason": grade.reasoning,
                "mode": mode.mode,
            }
    except Exception as exc:  # noqa: BLE001 — the layer must never crash chat
        result["evidence_grade"] = {
            "grade": "missing",
            "reasoning": f"intent_aware_rag_layer_skipped: {exc!s}",
        }


# Back-compat underscore aliases.  agent_rag's _finalize_result calls
# these by their underscore names.
_apply_post_gen_validator = apply_post_gen_validator
_apply_intent_aware_rag_layer = apply_intent_aware_rag_layer


__all__ = [
    "apply_post_gen_validator",
    "apply_intent_aware_rag_layer",
    "_apply_post_gen_validator",
    "_apply_intent_aware_rag_layer",
]
