# No-read held-out RAG goldset protocol

> **Status**: PROTOCOL ONLY. No external-author held-out evaluation has
> been completed against this protocol yet. Anything in
> `Data/evals/rag/latest_rag_holdout_baseline_comparison.json` should
> be read as `completed: false` until a reviewer has authored cases
> under the rules below.

## Purpose

The internal frozen goldset
(`Data/evals/rag/retrieval_goldset.jsonl`, 74 cases) has been used —
directly and indirectly — to:

- pick retrieval configurations,
- tune `LOGICAL_SOURCE_ALIASES`,
- write failure-analysis taxonomies,
- shape eval thresholds.

Even when those uses were not "training" in a ML sense, the goldset
has been *seen* by every engineering decision so far. Its result is
**in-sample**, not held-out. Any number computed against it overstates
how the system will behave on cases nobody on the project has read.

This protocol prepares a **held-out** retrieval goldset authored by a
reviewer who has **not read** the internal goldset, the alias map,
the failure analysis, or any of the existing RAG prompts. The result
on that held-out set is the closest this project can get to external
validation under its hard constraints (synthetic-only, no clinician,
no IRB).

## Why internal goldsets are contamination-prone

- Authoring an internal goldset case often involves reading the KB to
  decide which `expected_source_id`s belong there. The KB has been
  shaped by the same people who later tune the retriever.
- Failure analysis on the internal goldset names exact failure
  categories. Anyone reading those is no longer naive when authoring
  new cases.
- Source-alias additions are made *because* a specific case failed
  on a specific alias. Future cases authored by someone who has seen
  that alias map will subconsciously avoid the same failure shape.

The held-out set is meant to break that loop.

## Author eligibility

The author of a held-out case **must not** have read, before
authoring:

1. `Data/evals/rag/retrieval_goldset.jsonl` (the existing 74 cases)
2. `Data/evals/rag/latest_rag_baseline_comparison.json` (current
   configuration scores)
3. `Data/evals/rag/latest_rag_baseline_failures.json`
4. `Data/evals/rag/latest_retrieval_failure_analysis.json`
5. `Data/evals/rag/latest_source_alias_coverage.json`
6. `backend/services/rag_baseline_comparison.LOGICAL_SOURCE_ALIASES`
7. The deterministic refusal templates and intent-routing prompts in
   `backend/services/`
8. Any of the existing RAG-related ADRs
   (`docs/adr/0001-`, `0005-`, `0009-` in particular)

If the author has read **any** of those, the case is internal-
authored, not held-out — file it under the internal goldset, not
this template.

## Recommended reviewer roles

- **A classmate / peer engineer** (cheapest, fastest, most likely to
  be available). Authors easy/hard education + portal-help variants.
- **A senior AI / MLE reviewer** unaffiliated with this project.
  Authors hard contradiction + no-evidence + Taglish cases.
- **A healthcare-aware reviewer** (med student, nurse trainee, or
  similar). Authors urgent-symptom + supplement + tumor-marker
  variants where the safety-vs-education boundary matters most.
- **A clinician or genetic counselor**, if available. Authors
  genetics/VUS cases. This is the highest-value reviewer the
  project can engage; their time should be reserved for VUS and
  genetic-counseling categories.

Reviewer pairing is OK; co-authoring is OK. Anonymous reviewer
identity is OK — the author field accepts a role descriptor
(`external_peer_engineer`, `external_nurse_trainee`, etc.) rather
than a name.

## Case authoring instructions

1. Open the template file
   `Data/evals/rag/retrieval_goldset_holdout_v2_template.jsonl` in a
   plain text editor. Each line is one case.
2. Without consulting the KB, write a `query` you (as a reasonable
   patient or peer) would expect the agent to answer.
3. Without consulting the existing alias map, pick the
   `expected_source_ids` that you would *expect* a source-governed
   RAG to use. Use human-readable canonical labels:
   `infection-safety`, `genetic-counseling`, `tumor-marker-context`,
   `portal-help`, etc. Do not invent hashed IDs.
4. Decide the `expected_intent` and `expected_answerability_status`
   from the live enums:
   - `intent`: `education`, `portal_help`, `record_explanation`,
     `clinician_context`, `safety_routing`, `insufficient_evidence`.
   - `answerability_status`: `answerable_with_citations`,
     `answerable_with_limited_context`, `insufficient_evidence`,
     `conflicting_evidence`, `clinician_review_required`,
     `refuse_due_to_safety`.
5. Mark `expected_refusal_or_escalation: true` for urgent-symptom and
   safety-routing cases.
6. Add at least one `contradiction_trap` — a wrong patient-facing
   claim the agent must NOT produce (e.g. "VUS means positive",
   "CA 15-3 proves recurrence").
7. Set `authored_by` to your reviewer role descriptor and
   `internal_vs_external_authored: "external"`.
8. Set `was_used_for_tuning: false` and **leave it false**. If you
   later use the holdout case to debug, mark it `true` and the case
   is no longer held-out.

## Required fields

Every held-out case must include all fields the runner reads:

| Field | Type | Notes |
|---|---|---|
| `case_id` | string | `retrieval_holdout_v2_NNN` |
| `query` | string | Patient or peer-style natural language |
| `user_query` | string | Same as `query` (mirrors internal schema) |
| `category` | enum | `easy_education`, `hard_contradiction`, `no_evidence`, `taglish`, `genetics_vus`, `tumor_marker`, `supplement`, `urgent_symptom`, `source_tier_filtering` |
| `expected_intent` | enum | from the list above |
| `expected_answerability_status` | enum | from the list above |
| `expected_refusal_or_escalation` | bool | true for urgent / safety |
| `expected_refusal_or_insufficient_evidence` | bool | true for `no_evidence` |
| `expected_source_ids` | string[] | Human-readable canonicals only |
| `expected_allowed_use` | string | e.g. `general_patient_education` |
| `acceptable_source_tiers` | string[] | subset of `T1`/`T2`/`T3`/`T4`/`T5` |
| `required_source_tiers` | string[] | subset of acceptable |
| `contradiction_traps` | string[] | wrong claims the agent must not produce |
| `pass_criteria` | string[] | per-case acceptance rules |
| `fail_criteria` | string[] | per-case rejection rules |
| `authored_by` | string | reviewer role descriptor |
| `authored_date` | ISO date | `YYYY-MM-DD` |
| `internal_vs_external_authored` | string | always `"external"` |
| `was_used_for_tuning` | bool | always `false` |
| `case_source` | string | `external_author_no_read_protocol_v2` |
| `clinical_validation` | bool | always `false` |
| `safety_notes` | string | "Engineering retrieval/grounding test only. Not clinician-reviewed and not clinical validation." |

The template already includes these fields with placeholder values
for each of the nine required categories.

## Contamination disclosure rules

- The author must declare which of the 8 author-eligibility documents
  they have NOT read. File it in
  `Data/evals/external_review/<reviewer_role>_<date>_attestation.md`
  before submitting cases.
- If the author cannot truthfully attest "none of the above",
  the cases are filed as internal-authored cases against the
  existing goldset, not as held-out v2.
- The repo owner files the attestation and the case file in the
  same commit. The next release-gate run includes both.

## Marking internal vs external

- Internal frozen goldset: `internal_vs_external_authored: "internal"`,
  `case_source: "engineering_authored_frozen_retrieval_case"`.
- Held-out v2: `internal_vs_external_authored: "external"`,
  `case_source: "external_author_no_read_protocol_v2"`.

The runner reads `internal_vs_external_authored` and refuses to
classify the holdout result as `completed: true` unless **all** cases
in `retrieval_goldset_holdout_v2.jsonl` carry `"external"`.

## Marking `was_used_for_tuning`

- Default: `false`. The runner refuses to classify the holdout result
  as `completed: true` if **any** case has `true`.
- If a case is ever used to debug retrieval (alias additions,
  threshold tweaks, prompt edits), the author MUST set
  `was_used_for_tuning: true`. That case is then no longer held-out
  and the artifact reports `completed: false` until at least 5
  remaining cases stay untainted.

## How to freeze the set

1. Author writes `Data/evals/rag/retrieval_goldset_holdout_v2.jsonl`
   following the template. Minimum 9 cases (one per required category).
2. Author commits the file with a commit message tagged
   `holdout-v2: <reviewer_role>`.
3. The repo owner runs
   `python scripts/run_rag_holdout_baseline_comparison.py` to compute
   the comparison and the artifact.
4. From this point, no PR may modify
   `retrieval_goldset_holdout_v2.jsonl` unless it carries a
   `holdout-v2-rotation:` commit message and links to a new
   reviewer attestation. The CI is *not* configured to enforce this —
   it is a social contract.

## What CANNOT be claimed even after completion

Even with a clean held-out result, the following remain untrue:

- The system is **not** clinically validated.
- The system has **not** received clinician sign-off.
- The system has **no** real patient data.
- The system has **no** IRB / ethics approval.
- The held-out evaluation does **not** establish real-world safety.
- The system has **not** demonstrated patient benefit.
- A high held-out recall does **not** prove the system is ready for
  deployment.
- The held-out result is an engineering signal, not a regulatory
  artifact.

These boundaries are unchanged by any score on the held-out set.

## Related artifacts

- Protocol artifact: `Data/evals/rag/latest_rag_holdout_baseline_comparison.json`
- Failures artifact: `Data/evals/rag/latest_rag_holdout_baseline_failures.json`
- Template: `Data/evals/rag/retrieval_goldset_holdout_v2_template.jsonl`
- Template README: `Data/evals/rag/retrieval_goldset_holdout_v2.README.md`
- Runner: `scripts/run_rag_holdout_baseline_comparison.py`
- ADR cross-references: [ADR 0005](../adr/0005-adversarial-holdout-variants.md), [ADR 0009](../adr/0009-source-alias-normalization.md)
