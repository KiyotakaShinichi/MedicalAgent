# RAG Retrieval Goldset

The retrieval goldset checks whether retrieval finds governed source evidence
before generation.

The current frozen retrieval set contains 74 internally authored cases. It is
larger and harder than the original 12-case seed set and includes easy
education, hard contradiction, no-evidence, Taglish, genetics/VUS, tumor-marker,
supplement, urgent symptom, and source-tier filtering cases.

Fields include:

- query
- category and category tags
- gold source IDs
- acceptable source tiers
- relevant chunk IDs when available
- contradiction traps
- near-duplicate distractors
- stale-source distractors
- clinician-only distractors
- expected allowed use
- expected answerability status
- expected refusal or insufficient-evidence behavior
- authored_by / authored_date / was_used_for_tuning
- contamination notes

Run:

```bash
python scripts/run_retrieval_goldset_eval.py
```

If the JSONL file already exists, the runner loads it as the frozen goldset
instead of rebuilding it from the older claim-grounding set. To intentionally
regenerate the frozen set, run:

```bash
python scripts/build_frozen_rag_goldset.py
```

Artifact:

```text
Data/evals/rag/latest_retrieval_goldset_eval.json
```

Do not claim cross-encoder improvement unless `improvement_proven` is true.
Current metrics are retrieval engineering evidence only, not medical correctness.

The eval reports alias-normalized source matching because some older gold IDs
name a governed source concept while the current KB indexes generated chunk
IDs, parent IDs, and source titles. Treat raw-ID mismatch separately from true
retrieval failure.

Failure analysis:

```bash
python scripts/run_retrieval_failure_analysis.py
```

Artifact:

```text
Data/evals/rag/latest_retrieval_failure_analysis.json
```

The failure artifact classifies issues as retrieval, metadata/source-ID
normalization, source governance, scoring, or goldset design before tuning.

## Stage-wise retrieval oracle diagnostic

[`backend/services/rag_stage_oracle_diagnostic.py`](../backend/services/rag_stage_oracle_diagnostic.py)
re-runs every retrieval stage in isolation and reports which stage
loses the gold source. It does **not** change retrieval ranking or
live-agent behaviour.

Current attribution on the 74-case internal goldset:

- corpus coverage = 1.0 — gold sources are all in the KB.
- BM25 / dense / hybrid candidate recall@50 = 0.9865.
- **source-tier filter retention = 0.8378** — the filter drops 16% of cases by design (clinician-only / stale / out-of-allowed-use chunks the patient-facing audience can't cite).
- citation-window retention = 0.8378 (window does not lose anything past the filter).
- oracle upper bound = 0.8378 vs actual full stack = 0.7838 → **oracle gap = 0.054**.

Top failure stage: **source_filter_drop (9 of 14 failures)**. The
bottleneck is source governance working as designed, not retrieval
quality. The remaining 5 failures split across rrf_ranking (2),
dense-only (1), sparse-only (1), and citation_window_drop (1).

See [`docs/evals/rag_baseline_comparison.md`](evals/rag_baseline_comparison.md)
for the full table.

## Citation-context pruner (eval-path experiment, NOT promoted)

A pruning layer
([`backend/services/citation_context_pruner.py`](../backend/services/citation_context_pruner.py))
has been added as a comparison configuration. It is **eval-path only**.

On the internal goldset it improved Recall@5 by +7.4pp but regressed
citation_precision by **−9.7pp** (0.5243 → 0.4275). Per the brief's
rule "do not optimize only for Recall@10 if citation precision
worsens", the pruner is not wired into the live agent.

See [`docs/evals/rag_baseline_comparison.md`](evals/rag_baseline_comparison.md)
for the full before/after table and the 12-category failure
breakdown in
[`Data/evals/rag/latest_citation_precision_failure_analysis.json`](../Data/evals/rag/latest_citation_precision_failure_analysis.json).

Source governance, source-tier filtering, refusal correctness, and
the post-generation validator remain non-negotiable in both
configurations. The pruner does not bypass any of them.

## Held-out / external-author v2 (PREPARED, NOT COMPLETED)

The 74-case set above is the **internal** frozen goldset. A separate
**held-out v2** set is prepared for external-author evaluation under a
no-read protocol but **has not been completed**:

- Protocol: [`docs/evals/no_read_rag_goldset_protocol.md`](evals/no_read_rag_goldset_protocol.md)
- Template: `Data/evals/rag/retrieval_goldset_holdout_v2_template.jsonl`
- Template README: `Data/evals/rag/retrieval_goldset_holdout_v2.README.md`
- Runner: `python scripts/run_rag_holdout_baseline_comparison.py`
- Artifact: `Data/evals/rag/latest_rag_holdout_baseline_comparison.json`

Until a reviewer who has NOT read the internal goldset, the alias
map, or the failure analyses authors cases under the protocol, the
artifact reports `completed: false` with
`status: "ready_for_external_authoring"`. The runner refuses to
fabricate an external result by reusing the internal goldset.

The internal goldset has shaped retrieval configuration choices and
the alias map. Its result is **in-sample**, not held-out, and the
current numbers do not prove raw retrieval superiority over BM25
(`improvement_proven_vs_bm25 = false`).
