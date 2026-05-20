"""Per-turn trace diagnostics builder.

Produces a single ``TurnTrace`` envelope summarizing the *reason* the
agent reached a given decision, intended for admin debug surfaces
and post-incident analysis.  This envelope is explicitly **not** a
chain-of-thought log — it stores discrete decision facts, not the
free-form reasoning that produced them.

What goes in here
~~~~~~~~~~~~~~~~~
- correlation_id, generated_at
- model_used (router_tier_model, answer_tier_model)
- safety_scope (level + scope + matched_safety_terms)
- intent (deterministic_intent, llm_intent, llm_confidence,
  llm_override_blocked, route_alternatives_considered)
- retrieval_summary (chunks retrieved, high-trust count,
  top_score, answerability_status, retrieval_confidence,
  source_tier_confidence, citation_support_confidence,
  evidence_conflict_flag, reason)
- emotional_distress (category, response_mode, matched_terms)
- post_gen_validator (decision, blocked_claim_types, escalated)
- refusal (refused, refusal_reason)
- compound_intent (segments + language hint, if present)
- latency_ms (stage breakdown the caller already has)

What is explicitly NOT stored
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- The LLM's free-form thinking / hidden chain-of-thought.
- The verbatim user message (kept in the regular eval log, not here).
- The verbatim final answer (also kept in the eval log).
- Any token-by-token model output.

If a contributor wants to add a field, they must classify it as a
*decision* or a *thought*.  Decisions are allowed.  Thoughts are not.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


# Whitelist of legal top-level keys in the trace envelope.  A future
# contributor cannot smuggle in a "thinking_text" key without
# explicitly extending this set (and the test that asserts it).
TURN_TRACE_TOP_LEVEL_KEYS: frozenset[str] = frozenset({
    "schema_version",
    "correlation_id",
    "generated_at",
    "model_used",
    "safety_scope",
    "intent",
    "retrieval_summary",
    "emotional_distress",
    "post_gen_validator",
    "refusal",
    "compound_intent",
    "latency_ms",
    "validator_latency_ms",
    "patient_id",
    "release_id",
})

# Explicit deny-list of substrings that would indicate a contributor
# is trying to log chain-of-thought.
COT_DENYLIST: tuple[str, ...] = (
    "thinking",
    "chain_of_thought",
    "cot",
    "reasoning_text",
    "internal_monologue",
    "draft_response",
    "scratchpad",
)


@dataclass
class TurnTrace:
    correlation_id: str
    generated_at: str
    model_used: dict[str, Any] = field(default_factory=dict)
    safety_scope: dict[str, Any] = field(default_factory=dict)
    intent: dict[str, Any] = field(default_factory=dict)
    retrieval_summary: dict[str, Any] = field(default_factory=dict)
    emotional_distress: dict[str, Any] = field(default_factory=dict)
    post_gen_validator: dict[str, Any] = field(default_factory=dict)
    refusal: dict[str, Any] = field(default_factory=dict)
    compound_intent: dict[str, Any] = field(default_factory=dict)
    latency_ms: dict[str, Any] = field(default_factory=dict)
    validator_latency_ms: float | None = None
    patient_id: str | None = None
    release_id: str | None = None
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "correlation_id": self.correlation_id,
            "generated_at": self.generated_at,
        }
        if self.model_used:
            payload["model_used"] = dict(self.model_used)
        if self.safety_scope:
            payload["safety_scope"] = dict(self.safety_scope)
        if self.intent:
            payload["intent"] = dict(self.intent)
        if self.retrieval_summary:
            payload["retrieval_summary"] = dict(self.retrieval_summary)
        if self.emotional_distress:
            payload["emotional_distress"] = dict(self.emotional_distress)
        if self.post_gen_validator:
            payload["post_gen_validator"] = dict(self.post_gen_validator)
        if self.refusal:
            payload["refusal"] = dict(self.refusal)
        if self.compound_intent:
            payload["compound_intent"] = dict(self.compound_intent)
        if self.latency_ms:
            payload["latency_ms"] = dict(self.latency_ms)
        if self.validator_latency_ms is not None:
            payload["validator_latency_ms"] = float(self.validator_latency_ms)
        if self.patient_id is not None:
            payload["patient_id"] = self.patient_id
        if self.release_id is not None:
            payload["release_id"] = self.release_id
        return payload


def _scrub_cot(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Strip any keys whose names match the chain-of-thought deny-list.

    Defense-in-depth: callers should never pass a CoT field, but if a
    future change accidentally does, we drop it here rather than letting
    it land in storage.
    """
    cleaned: dict[str, Any] = {}
    for k, v in payload.items():
        key_lower = k.lower()
        if any(token in key_lower for token in COT_DENYLIST):
            continue
        if isinstance(v, Mapping):
            cleaned[k] = _scrub_cot(v)
        else:
            cleaned[k] = v
    return cleaned


def build_turn_trace(
    *,
    correlation_id: str | None = None,
    model_used: Mapping[str, Any] | None = None,
    safety_scope: Mapping[str, Any] | None = None,
    intent: Mapping[str, Any] | None = None,
    retrieval_summary: Mapping[str, Any] | None = None,
    emotional_distress: Mapping[str, Any] | None = None,
    post_gen_validator: Mapping[str, Any] | None = None,
    refusal: Mapping[str, Any] | None = None,
    compound_intent: Mapping[str, Any] | None = None,
    latency_ms: Mapping[str, Any] | None = None,
    validator_latency_ms: float | None = None,
    patient_id: str | None = None,
    release_id: str | None = None,
) -> TurnTrace:
    """Construct a ``TurnTrace`` from the discrete decision-source dicts.

    Every input is run through ``_scrub_cot`` so a buggy caller can't
    sneak chain-of-thought through.
    """
    trace = TurnTrace(
        correlation_id=correlation_id or str(uuid.uuid4()),
        generated_at=datetime.now(timezone.utc).isoformat(),
        model_used=_scrub_cot(model_used or {}),
        safety_scope=_scrub_cot(safety_scope or {}),
        intent=_scrub_cot(intent or {}),
        retrieval_summary=_scrub_cot(retrieval_summary or {}),
        emotional_distress=_scrub_cot(emotional_distress or {}),
        post_gen_validator=_scrub_cot(post_gen_validator or {}),
        refusal=_scrub_cot(refusal or {}),
        compound_intent=_scrub_cot(compound_intent or {}),
        latency_ms=_scrub_cot(latency_ms or {}),
        validator_latency_ms=validator_latency_ms,
        patient_id=patient_id,
        release_id=release_id,
    )
    return trace


def validate_trace_payload(payload: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Return ``(ok, problems)`` for a trace payload about to be stored.

    Used by tests and by any future write path that wants to enforce
    the structural contract.  Problems contain keys that:

    * are not in ``TURN_TRACE_TOP_LEVEL_KEYS``, OR
    * recursively contain a CoT-suspect key name.
    """
    problems: list[str] = []
    for key in payload.keys():
        if key not in TURN_TRACE_TOP_LEVEL_KEYS:
            problems.append(f"unexpected_top_level_key:{key}")

    def _walk(obj: Any, path: str = "") -> None:
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                kl = str(k).lower()
                for tok in COT_DENYLIST:
                    if tok in kl:
                        problems.append(f"cot_suspect_key:{path}{k}")
                _walk(v, f"{path}{k}.")
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                _walk(item, f"{path}[{i}].")

    _walk(payload)
    return (not problems, problems)


__all__ = [
    "COT_DENYLIST",
    "TURN_TRACE_TOP_LEVEL_KEYS",
    "TurnTrace",
    "build_turn_trace",
    "validate_trace_payload",
]
