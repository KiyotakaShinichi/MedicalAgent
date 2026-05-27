# 0002 — Tier-aware LLM adjudication (70B router · 120B answer)

**Status**: accepted

## Context

Early builds used a single LLM tier for everything — intent routing,
retrieval reranking, claim adjudication, and the final answer.
That conflated **routing latency** (we want fast) with **answer
quality** (we want high). It also conflated cost.

## Decision

Two tiers, configured at call sites via a `tier` argument to
`_adjudicate_json(system, prompt, tier="router"|"answer")`:

- **Router tier** (`llama-3.3-70b-versatile`): intent routing, query
  rewriting, claim extraction, refusal-style drafting. Latency-bound.
- **Answer tier** (`openai/gpt-oss-120b`): final patient-facing answer
  composition; borderline-claim post-gen escalation. Quality-bound.

`FAST_MODE` (ADR 0003) short-circuits BOTH tiers; the choice of which
tier to call still flows through `_groq_json` even in FAST_MODE so the
call-site code is uniform.

## Consequences

- ✅ Median chat latency dropped ~12× (2,915ms → 231ms) because the
  router tier no longer waits on the 120B model.
- ✅ Cost-per-turn is now visible per tier in the cost-latency report.
- ⚠ Two model dependencies to monitor. The 120B model rate-limited
  during one earlier eval run; FAST_MODE is the documented escape.
- ⚠ The split is not enforced at type level — a future contributor
  could accidentally pass `tier="answer"` from the router path.

## Reversal cost

Low. Setting both tiers to the same model name reverts behaviour.
