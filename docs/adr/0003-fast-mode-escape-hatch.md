# 0003 — FAST_MODE escape hatch + admin runtime toggle

**Status**: accepted

## Context

Three failure modes had previously taken the chat surface down:
1. Groq rate limits returning 429 mid-response.
2. Long Ollama HTTP timeouts on the router path.
3. Test runs accidentally hitting the live LLM (slow + non-hermetic).

We needed one switch that disables every LLM call in the agent stack
without code changes — usable from tests, the admin UI, and CI.

## Decision

Three-layer switch resolved by `local_llm.fast_mode_enabled()`:

1. **Env var** `ONCOTRACK_FAST_MODE=1` — set in CI and by the
   adversarial regression runner.
2. **Runtime override** via `local_llm.set_fast_mode_override(True|False|None)`
   — used by `/admin/fast-mode`.
3. **Default** off in production, on in tests via `conftest.py`.

When FAST_MODE is on, every `_adjudicate_json` call short-circuits and
returns the deterministic fallback envelope.

## Consequences

- ✅ Hermetic tests by default. The adversarial regression bank runs
  in <10s with FAST_MODE forced on.
- ✅ The admin UI exposes Force ON / Force OFF / Clear override so
  ops can toggle without redeploy.
- ✅ Rate-limit incidents now have a single mitigation path.
- ⚠ Anything gated on the LLM router's verdict silently degrades to
  the deterministic branch under FAST_MODE — that's the *intent*, but
  it means FAST_MODE itself becomes a source of behavioural difference
  that needs to be reflected in any A/B comparison.

## Reversal cost

Trivial. Delete the env var + override + admin endpoint.
