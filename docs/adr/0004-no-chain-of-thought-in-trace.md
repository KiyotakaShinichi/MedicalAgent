# 0004 — Per-turn trace stores decisions, never chain-of-thought

**Status**: accepted

## Context

The per-turn trace envelope (PART 6 of the external-critique pass) is
designed for admin debug surfaces and post-incident review. The
external critique flagged storing the LLM's free-form reasoning,
draft answers, or scratchpads as both a privacy risk (it leaks model
intent) and a hallucination amplifier (people read drafts as facts).

## Decision

Decisions, not thoughts.

1. `TURN_TRACE_TOP_LEVEL_KEYS` is an **explicit whitelist** of the 16
   legal top-level keys. A future contributor cannot add a new key
   without touching the whitelist *and* the test that asserts it.
2. `COT_DENYLIST` lists 7 forbidden substrings: `thinking`,
   `chain_of_thought`, `cot`, `reasoning_text`, `internal_monologue`,
   `draft_response`, `scratchpad`.
3. `_scrub_cot()` walks every input dict at construction time and
   drops keys matching the deny-list, recursively. Defense in depth.
4. `validate_trace_payload()` is available at any future write path
   to assert the contract before persistence.

## Consequences

- ✅ A contributor who passes `{"thinking": "..."}` accidentally loses
   the field silently — no error spam, but the data does not land in
   storage.
- ✅ The test `test_agent_turn_trace.py::ScrubChainOfThought` makes
   the deny-list completeness a CI-gated invariant.
- ⚠ Extending the trace requires both an ADR-style decision and an
   explicit whitelist update. That's intentional friction.

## What this is NOT

This ADR does not claim the LLM never produces chain-of-thought —
only that the agent **does not persist it** in the trace envelope.
The chain-of-thought may still exist in transient memory during a
single turn; it just never reaches the eval log.

## Reversal cost

High in terms of trust. Trivial in terms of code. Don't reverse
without a documented privacy / safety review.
