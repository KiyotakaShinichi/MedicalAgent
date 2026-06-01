# Negative results gallery

> **Listing a negative result here does NOT retroactively make any
> system component clinically valid.** This document is engineering
> credibility scaffolding. Not clinical validation.

The artifact at
[`Data/evals/governance/latest_negative_results_gallery.json`](../Data/evals/governance/latest_negative_results_gallery.json)
is the machine-readable source of truth. This doc is the human
reading version.

## Why this gallery exists

Most projects under-report negatives. The credibility risk of doing
that under this project's hard constraints (synthetic-only, no
clinician, no IRB) is much larger than the embarrassment of saying
"this didn't work". The gallery makes every honest negative finding
already documented elsewhere in the repo visible in one place.

## Catalogue

Each entry carries: `title`, `evidence_artifact` (path), `metric_value`,
`why_it_matters`, `decision_taken`, `what_was_not_claimed`,
`next_action`, `clinical_validation: false`.

1. **Full source-governed RAG stack does not beat BM25 on raw Recall@10.**
   `improvement_proven_vs_bm25 = false`. Full stack: 0.7838, BM25: 0.8041.
   The stack earns its keep on safety/governance, not raw recall.
2. **Citation context pruner regressed citation precision** by −9.7pp
   (0.5243 → 0.4275). Pruner stays experimental; not wired into the
   live agent.
3. **Cross-encoder reranker lift not proven.** Kept off by default
   (`RAG_ENABLE_CROSS_ENCODER` is opt-in).
4. **Held-out adversarial generalisation is weak on the 4 hardened
   categories.** In-sample 1.0 vs held-out v1 ~0.0625 overall.
   The held-out result is informational only and the
   anti-contamination test enforces no held-out query appears in the
   original bank.
5. **Source_filter_drop is mostly a goldset/governance mismatch.**
   The dominant retrieval-failure stage is the patient-facing source
   filter doing what governance requires. 9/14 failures attributed
   here. Filter is NOT weakened; adjudication packet built.
6. **Toxicity signal is shortcut-prone.** Synthetic-generator
   structural leakage causes toxicity AUC ~1.0. Signal retained as
   monitor-only; treatment-influence promotion blocked.
7. **All synthetic ML metrics are engineering self-tests.** They
   describe the synthetic distribution; they do not establish
   clinical predictive validity. The synthetic data quality artifact
   self-labels as `synthetic_generator_quality_proxy` with a
   test-enforced disclaimer.
8. **External / no-read RAG holdout v2 is prepared but not completed.**
   The runner refuses to fake completion until a reviewer authors
   cases under the no-read protocol.
9. **Source_filter_drop adjudication packet is draft only.**
   `n_filled_decisions = 0` out of 9.
10. **No clinician review has been completed.** 5 review packets
    prepared, 0 filled attestations.

## What this gallery is NOT

- Not a regret list. Each negative is paired with the decision the
  project owner took and the next action they queued.
- Not a roadmap. The
  [10/10-under-constraints roadmap](ten_out_of_ten_under_constraints.md)
  is the roadmap; this gallery is the *attribution surface* the
  roadmap rests on.
- Not clinical validation. None of the negatives are clinical
  claims; none of the decisions are clinical decisions.

## Related

- [`docs/ten_out_of_ten_under_constraints.md`](ten_out_of_ten_under_constraints.md)
- [`docs/evals/rag_baseline_comparison.md`](evals/rag_baseline_comparison.md)
- [`docs/evals/rag_goldset_adjudication.md`](evals/rag_goldset_adjudication.md)
- [`docs/adr/0005-adversarial-holdout-variants.md`](adr/0005-adversarial-holdout-variants.md)
- [`docs/adr/0009-source-alias-normalization.md`](adr/0009-source-alias-normalization.md)
