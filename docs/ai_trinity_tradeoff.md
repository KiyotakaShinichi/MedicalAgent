# AI Trinity: Accuracy, Latency, and Unit Cost

NLCare treats Accuracy, Latency, and Unit Cost as a non-compensatory engineering gate. A cheaper or faster route cannot be promoted when source governance, refusal correctness, citation grounding, or unsupported-context limits fail.

## Decision order

1. Safety and source governance must pass.
2. Accuracy and grounding must pass.
3. Latency must fit the declared internal budget.
4. Unit cost must be supported by provider-reported token telemetry and fit the planning budget.

The order matters. NLCare does not use one weighted score that lets a large latency gain hide unsafe leakage or weak grounding.

## Accuracy axis

Retrieval candidates are compared on the same internal frozen goldset using Recall@10, MRR, nDCG@10, citation precision, claim-support rate, unsupported-context rate, refusal correctness, and source-tier correctness. The composite accuracy-grounding score helps rank candidates, but the individual safety and quality floors remain binding.

## Latency axis

Retrieval p95 and route p95 are reported separately. Local percentiles are engineering measurements, not production SLO evidence. Cold-start, provider, database, network, and cache conditions must remain visible rather than being merged into one favorable number.

## Unit-cost axis

The preferred measure is:

`provider-token-derived cost / safe, source-governed, claim-supported answers`

Provider usage coverage must be at least 80% with at least 30 paired requests before the cost axis is complete. Character-count estimates and planning price scenarios remain separate from provider-reported usage. Missing provider telemetry is `unknown`, never `$0`.

## Current interpretation

The source-governed stack remains the operating policy because it preserves source-tier and refusal correctness. It is not presented as the raw retrieval, latency, or unit-cost winner. Promotion remains blocked while grounding floors fail or provider cost evidence is incomplete.

This is synthetic/internal engineering governance. It is not clinical validation, audited billing, a production SLO, or evidence of patient benefit.
