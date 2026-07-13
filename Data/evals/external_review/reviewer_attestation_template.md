# Reviewer attestation

> **One attestation per reviewer per engagement.** Commit alongside
> the matching intake form (`reviewer_intake_template.md`) and the
> feedback artifact (CSV or packet markdown).

---

## Reviewer

- **reviewer_role**: `_____________________________________________________________`
- **engagement_date**: `__________` *(YYYY-MM-DD)*
- **engagement_type** *(one of: held_out_rag_authoring /
  source_filter_drop_adjudication / adversarial_case_authoring /
  clinician_safety_review / genetic_counselor_vus_review /
  senior_mle_review / usability_review)*:
  `_____________________________________________________________`

## Contamination disclosure (required)

I attest that, **before** I authored / reviewed the artifact named
in `engagement_type`, I had NOT read any of the following:

- [ ] `Data/evals/rag/retrieval_goldset.jsonl`
- [ ] `Data/evals/rag/latest_rag_baseline_comparison.json`
- [ ] `Data/evals/rag/latest_rag_baseline_failures.json`
- [ ] `Data/evals/rag/latest_retrieval_failure_analysis.json`
- [ ] `Data/evals/rag/latest_source_alias_coverage.json`
- [ ] `Data/evals/rag/latest_citation_precision_failure_analysis.json`
- [ ] `Data/evals/rag/latest_rag_stage_oracle_diagnostic.json`
- [ ] `backend/services/rag_baseline_comparison.LOGICAL_SOURCE_ALIASES`
- [ ] The deterministic refusal templates in `backend/services/`
- [ ] Any RAG-related ADR (`docs/adr/0001-`, `0005-`, `0009-`)

If I cannot truthfully tick all items above for a held-out
RAG / adversarial authoring engagement, this engagement is filed as
**advisory** (not held-out), and any cases I authored are filed as
**internal**, not as a held-out set.

## Boundary acknowledgements (required)

- [ ] My review does not constitute clinical approval, clinician sign-off,
      IRB clearance, or any form of clinical validation.
- [ ] My role descriptor will be recorded in the repo; my real name
      will not be requested.
- [ ] I will not be paid for this review unless a separate explicit
      agreement is in place.
- [ ] I understand the project owner may either address my comments
      in a follow-up PR or mark them `wont_fix_with_rationale`;
      none will be silently dropped.

## Notes

Free-form notes for the reviewer (optional). Anything that would
help a future reviewer of the same artifact understand the context
of this engagement.

```
< add notes here >
```

## Filing

Save this file as
`Data/evals/external_review/<role>_<date>_attestation.md` and commit
it alongside the intake form and the feedback artifact.

The attestation is the **single artifact** the release-gate readiness
script looks at to count a review toward `completed_reviews`. No
intake form without a matching attestation will be counted.

## Anti-fabrication rule

The repo owner attests that this template, the intake template, the
feedback CSV, and the outreach message templates have NOT been
filled by the repo owner with synthetic / fabricated reviewer
identities. Any future commit that does so is treated as a
release-gate failure and must be reverted under the contamination-
disclosure policy.
