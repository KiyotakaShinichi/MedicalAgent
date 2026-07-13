# Review execution checklist

> Six engagement workflows, one section each. Each section names
> exactly what the reviewer may read, what they must not read
> before authoring, the files they fill, the validator command, the
> output artifact, and the bounded claims the engagement permits.

The repo owner runs
`python scripts/run_external_review_execution_readiness.py` after
each engagement lands to refresh the readiness artifact. No engagement
is auto-counted as completed.

---

## 1. Held-out RAG author (`held_out_rag_authoring`)

**May read**:

- `docs/evals/no_read_rag_goldset_protocol.md`
- `Data/evals/rag/retrieval_goldset_holdout_v2.README.md`
- `Data/evals/rag/retrieval_goldset_holdout_v2_template.jsonl`

**Must NOT read before authoring**:

- `Data/evals/rag/retrieval_goldset.jsonl`
- `Data/evals/rag/latest_rag_baseline_comparison.json`
- `Data/evals/rag/latest_rag_baseline_failures.json`
- `Data/evals/rag/latest_retrieval_failure_analysis.json`
- `Data/evals/rag/latest_source_alias_coverage.json`
- `Data/evals/rag/latest_rag_stage_oracle_diagnostic.json`
- `backend/services/rag_baseline_comparison.LOGICAL_SOURCE_ALIASES`
- RAG-related ADRs (`0001`, `0005`, `0009`)

**Files to fill**:

- Copy `retrieval_goldset_holdout_v2_template.jsonl` to
  `Data/evals/rag/retrieval_goldset_holdout_v2.jsonl` and add cases
  (minimum 9, recommended 15–30).
- `Data/evals/external_review/<role>_<date>_intake.md`
- `Data/evals/external_review/<role>_<date>_attestation.md`

**Validate**:

```
python scripts/run_rag_holdout_baseline_comparison.py
```

**Output artifact**:

- `Data/evals/rag/latest_rag_holdout_baseline_comparison.json`
  (status flips from `ready_for_external_authoring` → real
  comparison once gates pass).

**Allowed claim after completion**:

> "Held-out v2 RAG comparison authored by external reviewer under
> no-read protocol. Engineering generalisation signal only."

**Not allowed after completion**:

> Clinical validation, retrieval-quality lift over real-world
> baselines, regulatory readiness.

---

## 2. Source-filter-drop adjudicator (`source_filter_drop_adjudication`)

**May read**:

- `docs/evals/rag_goldset_adjudication.md`
- `Data/evals/rag/source_filter_drop_adjudication_packet.json`
  (draft state)

**Must NOT read before adjudicating**:

- The full goldset (`retrieval_goldset.jsonl`)
- The alias map in `LOGICAL_SOURCE_ALIASES`
- Stage-wise diagnostic JSON

**Files to fill**:

- The packet (decide per-item:
  `keep_expected_sources | revise_patient_facing_expected_sources |
  move_to_clinician_facing_goldset |
  split_patient_and_clinician_cases |
  mark_ambiguous_needs_external_review`).
- Intake + attestation.

**Validate**:

```
python scripts/validate_rag_goldset_adjudication.py
```

**Output artifact**:

- Same packet, with filled `reviewer_decision`, `reviewer_role`,
  `reviewer_notes`. Readiness JSON refreshes.

**Allowed claim after completion**:

> "Goldset adjudication completed for the 9 source-filter-drop
> cases by external reviewer. Engineering goldset-labelling
> correction only."

**Not allowed**:

> Clinical correctness, source governance weakening.

---

## 3. Adversarial safety case author (`adversarial_case_authoring`)

**May read**:

- `docs/review_packets/external_author_eval_packet.md`
- `Data/evals/safety/external_author_adversarial_template.jsonl`

**Must NOT read before authoring**:

- `Data/evals/safety/adversarial_safety_regression_bank.jsonl`
- `Data/evals/safety/adversarial_safety_holdout_variants.jsonl`
- `Data/evals/safety/latest_adversarial_*` artifacts
- `backend/services/agent_safety.py` vocabulary tables

**Files to fill**:

- Add cases to the external-author template (minimum 15).
- Intake + attestation.

**Validate**:

```
python scripts/run_adversarial_safety_regression.py
```

**Output artifact**:

- A new external-author held-out adversarial JSON next to
  `latest_adversarial_safety_holdout.json`.

**Allowed claim after completion**:

> "External-author adversarial cases (N) authored under attestation.
> Engineering safety-regression signal only."

**Not allowed**:

> Clinical safety validation, real-world adversarial robustness.

---

## 4. Clinician / oncology nurse safety reviewer (`clinician_safety_review`)

**May read**:

- `docs/review_packets/nurse_or_clinician_safety_review_packet.md`
- Patient-facing refusal templates + urgent-symptom triggers (in
  the packet).

**Must NOT review**:

- Any system component as a clinical sign-off.

**Files to fill**:

- Inline mark-up of the packet.
- Intake + attestation.
- Feedback CSV row(s).

**Validate** (manual):

The repo owner reviews the markup, files each comment in the
appropriate code/template/doc location, and links each to a commit
or `wont_fix_with_rationale`.

**Output artifact**:

- `Data/evals/external_review/<role>_<date>_clinician_safety_review.md`
- Updated `medical_safety_boundaries` evidence in the
  10/10-under-constraints roadmap.

**Allowed claim after completion**:

> "Patient-facing refusal templates reviewed by an oncology
> nurse/resident for wording. Engineering wording review only."

**Not allowed**:

> Clinical approval, clinical sign-off, validation of any specific
> system output, treatment authority.

---

## 5. Genetic counselor VUS reviewer (`genetic_counselor_vus_review`)

**May read**:

- `docs/review_packets/genetic_counselor_vus_review_packet.md`
- VUS handling code references inside the packet.

**Must NOT review**:

- The genetic_counseling_readiness ML signal as clinical guidance.

**Files to fill**:

- Inline mark-up of the packet.
- Intake + attestation.
- Feedback CSV row(s).

**Validate** (manual): same as clinician review.

**Output artifact**:

- `Data/evals/external_review/<role>_<date>_genetic_counselor_vus_review.md`
- Updated `medical_safety_boundaries` evidence in the roadmap.

**Allowed claim after completion**:

> "VUS patient-facing wording reviewed by a genetic counselor.
> Engineering wording review only."

**Not allowed**:

> Genetic interpretation validity, clinical risk advice,
> patient-specific guidance.

---

## 6. Senior MLE reviewer (`senior_mle_review`)

**May read**:

- `docs/review_packets/senior_mle_eval_review_packet.md`
- Leakage audit, patient-temporal CV, conformal calibration,
  shortcut audit, subgroup metrics, ML statistical evidence
  artifacts (all referenced in the packet).

**Must NOT treat as**:

- Clinical performance evidence.

**Files to fill**:

- Written feedback markdown or CSV rows.
- Intake + attestation.

**Validate** (manual): the repo owner files each comment in a
follow-up PR or marks `wont_fix_with_rationale`.

**Output artifact**:

- `Data/evals/external_review/<role>_<date>_senior_mle_review.md`
- Updated `ml_mle_engineering` + `ml_statistical_rigor` evidence in
  the roadmap.

**Allowed claim after completion**:

> "Senior MLE engineering review completed by external reviewer.
> Engineering review only."

**Not allowed**:

> Clinical predictive validity, deployment readiness, regulatory
> readiness.

---

## After ANY engagement

1. Verify `reviewer_intake_template.md`, `reviewer_attestation_template.md`,
   and (CSV or packet markdown) are all committed.
2. Run:
   ```
   python scripts/run_external_review_execution_readiness.py
   ```
3. Run release gate:
   ```
   python scripts/run_release_gate.py
   ```
4. Update the
   [10/10-under-constraints roadmap](../ten_out_of_ten_under_constraints.md)
   only if the underlying dimension's evidence actually changed.
   The readiness artifact is **not** sufficient to move scores —
   only the substance of the review is.
