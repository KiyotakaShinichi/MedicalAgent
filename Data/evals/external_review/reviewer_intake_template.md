# Reviewer intake form

> **Fill ONE intake form per reviewer per review session.** Commit it
> alongside the reviewer's feedback. Role descriptor is required;
> real name is optional and not requested.

---

## Reviewer

- **reviewer_role** *(required, role descriptor only, e.g.
  "oncology nurse, NN years experience", "external peer engineer",
  "board-certified genetic counselor")*:
  `_____________________________________________________________`
- **reviewer_background_summary** *(required, ≤ 200 chars,
  one-line professional summary without identifying details)*:
  `_____________________________________________________________`
- **date** *(required, YYYY-MM-DD)*:
  `__________`

## Engagement scope

- **artifact_reviewed** *(required, repo-relative path to the
  packet or eval set the reviewer engaged with)*:
  `_____________________________________________________________`
- **cases_reviewed** *(optional, list of case_ids or page numbers)*:
  `_____________________________________________________________`
- **time_spent_minutes** *(required, integer)*:
  `___`
- **conducted_under_no_read_protocol** *(required, yes/no — required
  for held-out RAG goldset + adversarial case authoring; otherwise
  "n/a")*:
  `_____`

## Contamination disclosure

The reviewer attests they have NOT read (before authoring or
reviewing) the following contamination-prone files:

- [ ] `Data/evals/rag/retrieval_goldset.jsonl`
- [ ] `Data/evals/rag/latest_rag_baseline_comparison.json`
- [ ] `Data/evals/rag/latest_rag_baseline_failures.json`
- [ ] `Data/evals/rag/latest_retrieval_failure_analysis.json`
- [ ] `Data/evals/rag/latest_source_alias_coverage.json`
- [ ] `backend/services/rag_baseline_comparison.LOGICAL_SOURCE_ALIASES`
- [ ] The deterministic refusal templates in `backend/services/`
- [ ] Any RAG ADR (0001, 0005, 0009)

If the reviewer cannot truthfully tick all of the above for a held-
out RAG / adversarial authoring engagement, the resulting cases are
filed as **internal** (not held-out) and the review is filed as
**advisory** (not held-out).

## Boundary acknowledgement

- [ ] *(required)* I understand this review **does not constitute
      clinical approval, clinician sign-off, IRB clearance, or any
      form of clinical validation.**
- [ ] *(required)* I understand my role descriptor will be
      acknowledged in the repo; my real name will not be requested
      or recorded.
- [ ] *(required)* I understand the project owner may not act on
      every comment; comments will be either addressed in a follow-
      up PR or marked `wont_fix_with_rationale` — none will be
      silently dropped.

## What the project owner commits to

- All comments will be filed in the corresponding artifact (review
  packet, adjudication packet, or feedback CSV) within 7 days.
- No comment will be silently dropped.
- The reviewer's role descriptor will be acknowledged in the
  next release-gate run that consumes their feedback.
- No claim of clinical validation or clinician approval will be
  made on the basis of this review.

## How to file this intake

1. Save this file as
   `Data/evals/external_review/<role>_<date>_intake.md` with the
   placeholders filled.
2. Commit it alongside the matching
   `<role>_<date>_attestation.md` (template:
   `reviewer_attestation_template.md`) and the feedback file
   (CSV or packet-specific markdown).
3. Run
   `python scripts/run_external_review_execution_readiness.py`
   to refresh the readiness artifact.

The runner does not auto-promote any review to `completed`; that
requires a separate reviewer-attested PR.
