# Paired RAG Statistical Comparison

This report uses the case pairing already present in the frozen internal RAG baseline comparison. It does not change retrieval or tune the goldset.

## raw_hybrid_rewrite_vs_bm25

- Scope: raw retrieval ablation; not the full governed stack
- Paired cases: `74`
- Favorable Recall@10 delta: `-0.087838`
- 95% paired bootstrap CI: `[-0.202703, 0.027027]`
- Holm-adjusted p-value: `0.706412`
- Improvement proven: `False`

## full_governed_stack_vs_bm25

- Scope: full source-governed stack versus sparse baseline
- Paired cases: `74`
- Favorable Recall@10 delta: `-0.155405`
- 95% paired bootstrap CI: `[-0.27027, -0.033784]`
- Holm-adjusted p-value: `0.072992`
- Improvement proven: `False`

## experimental_pruner_vs_full_governed_stack

- Scope: experimental pruner ablation; not a live-route promotion
- Paired cases: `74`
- Favorable Recall@10 delta: `0.060811`
- 95% paired bootstrap CI: `[-0.006757, 0.135135]`
- Holm-adjusted p-value: `1.0`
- Improvement proven: `False`

## Boundary

This is paired statistical evidence over one internally authored frozen engineering goldset. It does not establish clinical validation, external generalisation, patient benefit, or production healthcare readiness.
