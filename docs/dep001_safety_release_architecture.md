# DEP-001 Safety Release Architecture

## Layered Contract

1. Normalize current input and bounded prior user turns.
2. Apply deterministic urgent, treatment, diagnosis, privacy, and security
   checks.
3. Run the semantic unsafe-intent classifier as a separate signal.
4. For uncertain cases, compose requested-action and protected-target slots.
   Safety-critical uncertainty routes to refusal/clarification and disables
   cache reuse.
5. Run the security input guardrail before retrieval.
6. Route high-risk turns to deterministic refusal composition. Provider and
   caller fallback prose is discarded on that lane.
7. Apply source-tier and allowed-use policy to any educational retrieval.
   Retrieved text cannot change the safety envelope.
8. Run answer/citation validation, medical-claim boundary checks, the
   post-generation safety validator, and evidence-envelope authorization.
9. Buffer the response until release authorization. Exceptions at safety,
   retrieval, cache, generation, or validator layers return a fail-closed
   envelope.

The final holdout and evaluation implementation are never imported by these
production modules. This prevents rule logic from reading case text or IDs.

## Operational Boundary

The current release boundary is stronger than the current route classifier:
the frozen final bank released zero unsafe candidates but showed weak route,
urgent, multilingual, and multi-turn recall. DEP-001 therefore remains a hard
deployment blocker. The architecture is engineering evidence only, not
clinical validation or production-healthcare readiness.

