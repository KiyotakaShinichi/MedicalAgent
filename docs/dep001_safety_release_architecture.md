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
   post-generation safety validator, semantic output-actionability validation,
   and evidence-envelope authorization.
9. Buffer the response until release authorization. Exceptions at safety,
   retrieval, cache, generation, or validator layers return a fail-closed
   envelope.
10. Re-run semantic output-actionability validation immediately before transport.
    A post-generation/transport disagreement blocks release and strips citations.

The final holdout and evaluation implementation are never imported by these
production modules. This prevents rule logic from reading case text or IDs.

## Operational Boundary

The DEP-001D release boundary is stronger than the route classifier: its frozen
1,600-case blind bank released zero unsafe canaries and passed 10/10 fault
injections, but unsafe recall was 0.8725, safe acceptance was 0.7913, over-refusal
was 0.2088, and multi-turn unsafe recall was 0.8033. The one-shot result is
`BLOCKED_BEHAVIORAL`; candidate and bank are consumed. DEP-001 therefore remains
a hard deployment blocker. This is engineering evidence only, not clinical
validation or production-healthcare readiness.

Post-candidate development adds a narrowly bounded safe-utility consensus for
benign portal operations and direct clinical education. It may preserve a
low-risk route only when independent classifiers agree, unsafe and urgent
probabilities are low, explicit personalized-action grammar is absent, no
current urgent symptom is detected, and the independent urgent head is below
threshold. This is not part of the frozen DEP-001D candidate and has no new
blind-evaluation claim.
