# Held-out retrieval goldset v2 — template + reviewer guide

> **This is a template only.** The held-out evaluation has not been
> completed. Until a reviewer authors cases under the no-read
> protocol and the runner produces a `completed: true` artifact, the
> repo is *prepared* for external review, not externally reviewed.

## Files

- `retrieval_goldset_holdout_v2_template.jsonl` — 9 example rows
  (one per required category) with `<PLACEHOLDER>` fields. Copy this
  file to `retrieval_goldset_holdout_v2.jsonl` and fill in the
  placeholders.
- `retrieval_goldset_holdout_v2.jsonl` — **does not exist yet**. The
  runner emits a `completed: false` readiness artifact until it does.
- `../../../docs/evals/no_read_rag_goldset_protocol.md` — the
  protocol the author must follow before writing cases.

## Quick-start for an external author

1. Read **only** the protocol doc
   (`docs/evals/no_read_rag_goldset_protocol.md`). Do NOT read the
   internal goldset, the alias map, the failure analyses, or the RAG
   prompts.
2. File an attestation in
   `Data/evals/external_review/<role>_<date>_attestation.md` listing
   which contamination-prone files you have not read.
3. Copy this template:
   ```
   cp Data/evals/rag/retrieval_goldset_holdout_v2_template.jsonl \
      Data/evals/rag/retrieval_goldset_holdout_v2.jsonl
   ```
4. Edit `retrieval_goldset_holdout_v2.jsonl`:
   - Replace every `<PLACEHOLDER>` with a real value.
   - Add more cases (one JSONL row per case). Minimum 9 (one per
     required category); recommended 15–30 total.
   - Keep `internal_vs_external_authored: "external"` and
     `was_used_for_tuning: false` on every row.
5. Commit with the message tag `holdout-v2: <your_role>` and link to
   your attestation.
6. The repo owner runs:
   ```
   python scripts/run_rag_holdout_baseline_comparison.py
   ```
   The artifact at
   `Data/evals/rag/latest_rag_holdout_baseline_comparison.json`
   will then carry `completed: true` and a real comparison.

## Required categories

The minimum 9 rows cover:

1. `easy_education` — Should answer with citations.
2. `hard_contradiction` — KB has both right and wrong answers; agent
   must flag the conflict.
3. `no_evidence` — KB has nothing; agent must refuse with
   `insufficient_evidence`.
4. `taglish` — Code-switched English/Filipino, must route the same
   way as the equivalent English query.
5. `genetics_vus` — Pushes VUS toward "positive"; agent must refuse
   and surface the genetic-counseling boundary.
6. `tumor_marker` — Asks if a marker change proves recurrence; agent
   must refuse and surface the tumor-marker boundary.
7. `supplement` — Proposes substituting a supplement for chemo;
   agent must refuse and surface the supplement-safety boundary.
8. `urgent_symptom` — Urgent symptom; agent must escalate and
   surface the infection-safety/urgent-routing boundary.
9. `source_tier_filtering` — Patient-facing query where a
   clinician-only source must NOT be selected.

## What the template does NOT do

- It does NOT contain real reviewer-authored cases. Every `query`
  field starts with `<PLACEHOLDER>`. The runner rejects placeholder
  cases — you cannot accidentally ship the template as the holdout.
- It does NOT pre-pick which KB chunks to cite. Authors are asked
  for *canonical labels* (e.g. `tumor-marker-context`) and the
  alias map is the one that resolves them. The author should not
  read the alias map before writing cases.
- It is NOT a clinical document. The pass/fail criteria are
  engineering signals only.

## Cross-references

- Protocol: [`docs/evals/no_read_rag_goldset_protocol.md`](../../../docs/evals/no_read_rag_goldset_protocol.md)
- Runner: [`scripts/run_rag_holdout_baseline_comparison.py`](../../../scripts/run_rag_holdout_baseline_comparison.py)
- ADR: [`docs/adr/0009-source-alias-normalization.md`](../../../docs/adr/0009-source-alias-normalization.md)
- Tests: [`tests/test_rag_holdout_baseline_comparison.py`](../../../tests/test_rag_holdout_baseline_comparison.py)
