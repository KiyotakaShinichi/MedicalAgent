# Clinician-facing retrieval goldset — placeholder

> **No clinician-facing goldset exists yet.** This file is a
> placeholder. The corresponding JSONL file is intentionally NOT
> present.

## When this would be populated

If the adjudication workflow in
[`docs/evals/rag_goldset_adjudication.md`](../../docs/evals/rag_goldset_adjudication.md)
records any `move_to_clinician_facing_goldset` or
`split_patient_and_clinician_cases` decision, then a clinician-facing
JSONL would be added here. Until that happens, the patient-facing
goldset (`retrieval_goldset.jsonl`) is the only frozen set.

## What MUST be true if a clinician-facing goldset is added

1. **The file name is** `clinician_facing_retrieval_goldset.jsonl`
   and lives in this directory.
2. **Every case** must carry these fields explicitly:
   - `audience: "clinician"` (not `"patient"`).
   - `acceptable_source_tiers` may include `T4` / `T5` /
     `clinician_only` sources.
   - `expected_allowed_use` may include `clinician_only`,
     `medical_claim_boundary_or_insufficient_evidence`,
     `dose_adjustment_protocol`, etc.
   - `clinical_validation: false` — this is engineering scaffolding,
     not validated clinical content.
3. **Patient-facing metrics must NOT include clinician-only sources.**
   Any new comparison artifact that evaluates patient-facing
   retrieval against the clinician-facing goldset is rejected by the
   release gate.
4. **No automatic backflow.** A clinician-facing case may not migrate
   back to the patient goldset without a new reviewer attestation
   under the no-read protocol.

## What this is NOT

- **Not clinical validation.** A clinician-facing goldset is still
  engineering scaffolding — the labels are reviewer-authored, not
  clinically validated.
- **Not a substitute for clinician sign-off.** Having a
  clinician-facing eval set does not establish clinician approval of
  any system output.
- **Not a license to weaken the patient-facing source-tier filter.**
  The patient-facing filter exists to keep clinician-only sources out
  of patient-facing citations; that remains non-negotiable.

## Why a placeholder instead of an empty JSONL

A zero-byte JSONL would be ambiguous (was the file emptied? did the
build fail? did a reviewer accidentally commit a draft?). A README
that says "this does not exist yet" is the unambiguous state.

## Related

- Adjudication workflow:
  [`docs/evals/rag_goldset_adjudication.md`](../../docs/evals/rag_goldset_adjudication.md)
- Patient-facing goldset:
  [`retrieval_goldset.jsonl`](retrieval_goldset.jsonl)
- Held-out v2 protocol:
  [`docs/evals/no_read_rag_goldset_protocol.md`](../../docs/evals/no_read_rag_goldset_protocol.md)
- ADR 0005 (held-out variants):
  [`docs/adr/0005-adversarial-holdout-variants.md`](../../docs/adr/0005-adversarial-holdout-variants.md)
