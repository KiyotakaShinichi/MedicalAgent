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

    Optional 120B answer-tier escalation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    When the deterministic validator returns ``allowed`` but the reply
    contains "borderline" wording the rule set might miss, and the
    operator has opted into LLM escalation via
    ``ONCOTRACK_POSTGEN_ANSWER_ESCALATION=1``, we ask the 120B answer
    tier for a second opinion.  Same block behavior applies if it
    votes blocked.  FAST_MODE skips this escalation (it goes through
    the same ``_adjudicate_json`` short-circuit as every other LLM
    call).
    """
    from backend.services.post_generation_validator import validate_reply

    pgv_decision = validate_reply(result.get("reply") or "")
    if pgv_decision.decision == "blocked":
        _apply_block(result, output_guardrails, pgv_decision)
        # _apply_block can't mutate ``output_guardrails`` directly when
        # it needs to replace it; it returns the maybe-new dict via
        # result["_replaced_output_guardrails"] for the caller to use.
        output_guardrails = result.pop("_replaced_output_guardrails", output_guardrails)
        return output_guardrails, pgv_decision

    # Deterministic validator said "allowed".  Optional LLM second
    # opinion for borderline wording — gated by env var so the default
    # behavior is unchanged.
    escalated = _maybe_escalate_to_answer_tier(result)
    result["post_gen_validator"] = {
        "decision":               "allowed",
        "triggered_rules":        [],
        "medical_claim_boundary": pgv_decision.claim_boundary,
        "answer_tier_escalation": escalated,
    }
    if escalated and escalated.get("decision") == "blocked":
        # Build a synthetic pgv-like decision from the LLM verdict and
        # apply the same block path.
        from backend.services.post_generation_validator import validate_reply as _vr
        synthetic = type(pgv_decision)(  # reuse the dataclass shape
            decision="blocked",
            triggered_rules=tuple(escalated.get("triggered_rules") or ["llm_answer_tier_block"]),
            matched_excerpts=tuple(escalated.get("matched_excerpts") or []),
            suggested_response=escalated.get("suggested_response")
                or "I cannot safely answer that as stated. Please contact your oncology care team for medical review.",
            claim_boundary=pgv_decision.claim_boundary,
        ) if hasattr(pgv_decision, "__class__") else pgv_decision
        _apply_block(result, output_guardrails, synthetic)
        output_guardrails = result.pop("_replaced_output_guardrails", output_guardrails)
        return output_guardrails, synthetic

    return output_guardrails, pgv_decision


def _apply_block(result, output_guardrails, pgv_decision) -> None:
    """Apply the standard 'block' transformation to ``result`` in place
    and stash the (possibly replaced) ``output_guardrails`` dict under
    ``result["_replaced_output_guardrails"]`` for the caller to lift."""
    original_reply = result.get("reply")
    result["reply"] = pgv_decision.suggested_response
    result["citations"] = []
    result["post_gen_validator"] = {
        "decision":                "blocked",
        "triggered_rules":         list(pgv_decision.triggered_rules),
        "matched_excerpts":        list(pgv_decision.matched_excerpts),
        "medical_claim_boundary":  pgv_decision.claim_boundary,
        "original_reply_preview":  (original_reply or "")[:240],
    }
    if isinstance(output_guardrails, dict):
        replaced = dict(output_guardrails)
        replaced["status"] = "blocked_by_post_gen_validator"
        existing_issues = list(replaced.get("issues") or [])
        existing_issues.extend(
            f"post_gen::{rule}" for rule in pgv_decision.triggered_rules
        )
        replaced["issues"] = existing_issues
        result["_replaced_output_guardrails"] = replaced
    else:
        result["_replaced_output_guardrails"] = output_guardrails


# Borderline patterns: treatment-decision-adjacent / diagnosis-adjacent
# wording that the deterministic validator's strict pattern set might
# not catch.  Presence of one of these in an otherwise-allowed reply
# triggers the optional 120B second opinion.
_BORDERLINE_PATTERNS: tuple[str, ...] = (
    "in your case",
    "i think you",
    "i believe you",
    "this likely means",
    "this likely indicates",
    "this means that you",
    "you probably have",
    "you probably need",
    "would be safe",
    "should be safe",
    "it is fine to",
    "you can probably",
    "you can definitely",
    "no need to worry",
    "definitely not cancer",
    "definitely cancer",
)


def _maybe_escalate_to_answer_tier(result: dict[str, Any]) -> dict[str, Any] | None:
    """Optional second-opinion on a deterministic ``allowed`` verdict.

    Skipped unless ``ONCOTRACK_POSTGEN_ANSWER_ESCALATION=1`` AND the
    reply contains at least one borderline pattern.  Calls the 120B
    answer tier via ``_adjudicate_json(tier="answer")`` which itself
    honours ``ONCOTRACK_FAST_MODE`` and the per-tier model config.
    Returns the verdict dict, or None when the escalation didn't fire.
    """
    import os
    if os.environ.get("ONCOTRACK_POSTGEN_ANSWER_ESCALATION", "").strip().lower() not in {"1", "true", "yes"}:
        return None

    reply = (result.get("reply") or "").lower()
    matched = [p for p in _BORDERLINE_PATTERNS if p in reply]
    if not matched:
        return None

    try:
        from backend.services.local_llm import _adjudicate_json
        import json as _json
        verdict = _adjudicate_json(
            system=(
                "You are a strict post-generation safety reviewer for a non-diagnostic oncology "
                "support assistant. Return only JSON. Block any reply that diagnoses, predicts "
                "prognosis, recommends treatment, gives dosing, confirms genetic risk, concludes "
                "from tumor markers, or implies false reassurance. Allow safe education, "
                "record organization, and clinician-routing wording."
            ),
            prompt=_json.dumps({
                "task": "post_gen_borderline_review",
                "reply": result.get("reply"),
                "borderline_patterns_matched": matched,
                "return_json_schema": {
                    "decision":            "allowed | blocked",
                    "triggered_rules":     ["short strings"],
                    "matched_excerpts":    ["short strings"],
                    "suggested_response":  "short safe refusal if blocked",
                    "confidence":          "0.0-1.0",
                    "reason":              "short string",
                },
            }, ensure_ascii=False),
            tier="answer",
        )
    except Exception as exc:  # noqa: BLE001 — never crash chat on the second opinion
        return {"available": False, "reason": f"escalation_failed:{exc!s}"}

    if not verdict.get("available"):
        return verdict

    return {
        "available":         True,
        "decision":          "blocked" if verdict.get("decision") == "blocked" else "allowed",
        "triggered_rules":   verdict.get("triggered_rules") or [],
        "matched_excerpts":  verdict.get("matched_excerpts") or matched,
        "suggested_response": verdict.get("suggested_response"),
        "confidence":        float(verdict.get("confidence") or 0),
        "borderline_patterns_matched": matched,
        "model":             verdict.get("model"),
    }


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
        from backend.services.retrieval_confidence import classify_retrieval_uncertainty

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
        retrieval_confidence = classify_retrieval_uncertainty(
            chunks=filter_result.kept_chunks,
            claim_envelope=claim_validation.to_dict(),
            safety=result.get("safety") or input_guardrails or {},
            intent=result.get("intent") or mode.mode,
        )
        result["retrieval_confidence"] = retrieval_confidence.to_dict()

        # Substitute the mode's insufficient-evidence default when
        # grading collapses OR the answerability router says we must
        # not answer confidently, AND the validator hadn't already
        # blocked.  Accepted routing statuses for substitution:
        #   - insufficient_evidence
        #   - conflicting_evidence (would mislead if answered)
        #   - clinician_review_required (patient-specific + thin support)
        confidence_status = retrieval_confidence.answerability_status
        confidence_triggers_substitution = confidence_status in {
            "insufficient_evidence",
            "conflicting_evidence",
            "clinician_review_required",
        }
        should_substitute = (
            (grade.grade == "insufficient" or confidence_triggers_substitution)
            and pgv_decision.decision != "blocked"
            and mode.insufficient_evidence_default
        )
        if should_substitute:
            result["reply"] = mode.insufficient_evidence_default
            result["citations"] = []
            result["insufficient_evidence_substitution"] = {
                "reason": grade.reasoning if grade.grade == "insufficient" else retrieval_confidence.reason,
                "mode": mode.mode,
                "answerability_status": confidence_status,
                "evidence_conflict_flag": bool(retrieval_confidence.evidence_conflict_flag),
                "low_confidence_reason": (
                    retrieval_confidence.reason
                    if confidence_triggers_substitution
                    else None
                ),
                "trigger": (
                    "evidence_grade_insufficient"
                    if grade.grade == "insufficient"
                    else f"retrieval_confidence_{confidence_status}"
                ),
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
