# Eval contamination harmonisation

> **Pushing any eval artifact past its category — for instance, citing
> an `internal_used_for_tuning` number as "external generalisation"
> — is overclaiming, regardless of how green the metric looks.** This
> map does not change any artifact's content; it bounds the strongest
> reading each artifact deserves.

The machine-readable artifact is
[`Data/evals/governance/latest_eval_contamination_harmonization.json`](../../Data/evals/governance/latest_eval_contamination_harmonization.json).

## Why this exists

The repo carries 100+ eval artifacts spanning every stage of the
pipeline. Without an explicit harmonisation map, a reader can
honestly mistake an in-sample number for an external one, or treat a
synthetic metric as a clinical claim. The categories below are the
seven slots every eval artifact belongs to.

## Categories

| Category | What it means | Allowed claim strength |
|---|---|---|
| `internal_used_for_tuning` | The repo owner / engineering team has seen the artifact and has used it (directly or indirectly) to shape retrieval, alias map, prompts, or thresholds. | in-sample engineering signal only |
| `internal_frozen_not_used_for_tuning` | Authored internally but explicitly held back from tuning (e.g. adversarial held-out v1). | honest in-sample generalisation signal |
| `external_no_read_prepared_incomplete` | Templates / runners ready for an external author under the no-read protocol; no cases authored. | preparation only — does NOT establish external evidence |
| `external_completed` | External author has filed cases under attestation and the runner reports `completed: true`. | external generalisation signal (does NOT imply clinical validity) |
| `synthetic_generated` | The data itself is synthetic; metrics describe the synthetic distribution. | synthetic-only; NOT clinical evidence |
| `live_agent_internal` | Captured from live-agent behaviour in the engineering environment. | engineering behaviour signal only |
| `informational_only` | Diagnostic / categorisation / readiness artifact; does NOT carry a score claim. | engineering signal only |

## How this changes nothing on disk

- No eval artifact is modified.
- No release-gate threshold is changed.
- No live-agent behaviour is changed.
- The harmonisation map is a *vocabulary* the reader uses to interpret
  what they already see.

## How this changes everything in interpretation

- A reader who quotes `Data/evals/rag/latest_rag_baseline_comparison.json`
  Recall@10 as "the model's recall" is overclaiming: that artifact
  is `internal_used_for_tuning` and the strongest allowed reading is
  "in-sample comparison; `improvement_proven_vs_bm25 = false`".
- A reader who quotes the in-sample 1.0 adversarial bank rate as
  "the system blocks attacks" is overclaiming: that artifact is
  `internal_used_for_tuning` post-2026-05-20 hardening. The honest
  number is the held-out v1 result, which is
  `internal_frozen_not_used_for_tuning` and explicitly informational.
- A reader who quotes any synthetic ML metric as "model performance"
  is overclaiming: every synthetic artifact is `synthetic_generated`,
  and the audit footnote is required next to the number.

## Related

- [10/10-under-constraints roadmap](../ten_out_of_ten_under_constraints.md)
- [Negative results gallery](../negative_results_gallery.md)
- [Portfolio safe wording](../portfolio_safe_wording.md)
- [No-read RAG goldset protocol](no_read_rag_goldset_protocol.md)
- [RAG baseline comparison doc](rag_baseline_comparison.md)
