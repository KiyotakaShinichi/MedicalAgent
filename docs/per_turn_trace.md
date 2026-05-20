# Per-turn trace diagnostics

A single `TurnTrace` envelope summarizing **why** the agent reached a
given decision on one chat turn. Used by the admin debug surface and
for post-incident review.

## The hard rule: decisions, not thoughts

`TurnTrace` stores discrete decision facts. It does **not** store any
free-form chain-of-thought, hidden reasoning, internal monologue, or
draft response text. The module enforces this on two fronts:

1. `_scrub_cot` walks every input dict at construction time and drops
   keys whose names match the deny-list (`thinking`,
   `chain_of_thought`, `cot`, `reasoning_text`, `internal_monologue`,
   `draft_response`, `scratchpad`).
2. `validate_trace_payload` is available for any future write path
   that wants to assert the contract at storage time. It flags both
   unknown top-level keys and nested CoT-suspect names.

If a contributor adds a new field they have to (a) add the key to
`TURN_TRACE_TOP_LEVEL_KEYS` and (b) update
[`tests/test_agent_turn_trace.py`](../tests/test_agent_turn_trace.py)
to keep the lock-in green.

## Top-level shape

```jsonc
{
  "schema_version": "1.0",
  "correlation_id": "...",
  "generated_at": "...",
  "model_used": {"router_tier_model": "...", "answer_tier_model": "..."},
  "safety_scope": {"level": "...", "scope": "...", "matched_safety_terms": [...]},
  "intent": {
    "deterministic_intent": "...",
    "llm_intent": "...",
    "llm_confidence": 0.0,
    "llm_override_blocked": false,
    "route_alternatives_considered": [...]
  },
  "retrieval_summary": {
    "answerability_status": "...",
    "retrieval_confidence": 0.0,
    "source_tier_confidence": 0.0,
    "citation_support_confidence": 0.0,
    "evidence_conflict_flag": false,
    "top_k_evaluated": 0,
    "high_trust_chunks": 0,
    "reason": "..."
  },
  "emotional_distress": {"category": "...", "response_mode": "...", "matched_terms": [...]},
  "post_gen_validator": {"decision": "...", "blocked_claim_types": [...], "escalated": false},
  "refusal": {"refused": false, "refusal_reason": "..."},
  "compound_intent": { /* full envelope */ },
  "latency_ms": {"router": 0.0, "retrieval": 0.0, "generation": 0.0},
  "validator_latency_ms": 0.0,
  "patient_id": "...",
  "release_id": "..."
}
```

## Files

- Module: [`backend/services/agent_turn_trace.py`](../backend/services/agent_turn_trace.py)
- Tests: [`tests/test_agent_turn_trace.py`](../tests/test_agent_turn_trace.py)

## Wiring

The wiring into `agent_rag.run_patient_agent_pipeline` and the admin
trace surface in
`frontend-react/src/pages/admin/sections/AgentTraceSection.tsx` is a
follow-up. The module is shipped now so the schema and the no-CoT
contract are stable and tested.
