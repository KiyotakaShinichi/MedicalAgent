# 0001 — agent_rag.py god-module split into 15 focused modules

**Status**: accepted
**Date**: 2026-05 (during the stabilisation pass)

## Context

`backend/services/agent_rag.py` had grown to 2,076 lines and owned:
intent routing, safety-scope detection, retrieval, claim validation,
evidence grading, the post-gen validator, the cache, the eval log,
and the response finalizer. Six external call sites imported
helpers from it directly.

Symptoms: bug fixes touched unrelated layers; tests had to import the
whole stack; reviewers couldn't tell which layer a regression came
from.

## Decision

Split into 15 focused modules under `backend/services/` with stable,
single-purpose APIs. Keep `agent_rag.py` as a thin **dispatcher +
re-export shim** so the existing six call sites do not break.

Modules:
- `agent_safety` (safety_scope_check + vocabulary tables)
- `agent_intent_router` (route_intent + LLM augmenter)
- `agent_retrieval` (hybrid retrieval + rerank + compression)
- `agent_post_gen` (post-gen validator + intent-aware RAG layer)
- `rag_intent_modes`, `rag_tier_filter`, `rag_claim_validator`,
  `rag_evidence_grading`, `rag_intent_aware_eval`, `rag_tier_ablation`
- `compound_intent_router`, `multilingual_tool_router`
- `retrieval_confidence`, `agent_turn_trace`
- `agent_eval_log`

## Consequences

- ✅ Each layer is now unit-testable in isolation.
- ✅ The failure surface is explicit in `_finalize_result` — top-to-bottom.
- ✅ Re-export shim preserves the existing import surface; no big-bang
  rename in the call sites.
- ⚠ The dispatcher in `agent_rag.py` still owns orchestration; that's
  intentional — it's the *only* place that knows the full pipeline order.

## Reversal cost

High. Re-merging is mechanical but loses the layer separation.
Don't reverse unless a single layer needs to become an external service.
