# RAG goldset adjudication workflow

> **Status**: WORKFLOW ONLY. No adjudication decisions have been
> recorded yet. The packet is a draft — every item's
> `reviewer_decision` is `null`.

## Why this exists

The stage-wise retrieval oracle diagnostic
(`Data/evals/rag/latest_rag_stage_oracle_diagnostic.json`) attributes
the largest single failure stage on the frozen internal goldset to
**`source_filter_drop`** — 9 of 74 cases. The dropped chunks are
sources the patient-facing source-tier / allowed_use filter is
designed to exclude (clinician-only protocols, T4/T5 policy
documents, allowed_use=`medical_claim_boundary_or_insufficient_evidence`).

The correct response is **not to weaken the filter**. The brief is
explicit: source governance is non-negotiable. The right response is
to ask a reviewer whether each affected case should be:

1. **`keep_expected_sources`** — the goldset is right; the failure is
   a design tradeoff to record openly.
2. **`revise_patient_facing_expected_sources`** — the case is
   patient-facing; the gold list should be replaced with sources
   that pass the patient-facing filter.
3. **`move_to_clinician_facing_goldset`** — the case is
   clinician-facing and should not have been authored against the
   patient goldset to begin with.
4. **`split_patient_and_clinician_cases`** — the case has two valid
   interpretations and should be split into one patient-facing and
   one clinician-facing case.
5. **`mark_ambiguous_needs_external_review`** — none of the above;
   defer to external reviewer under the no-read protocol.

## Source_filter_drop is mostly a goldset/governance mismatch

Of the 9 drop cases:

- 7 expect `"Project safety policy"` at `acceptable_source_tiers: [T4]`.
  The patient-facing baseline filter excludes T4 by policy. The
  filter is **working as designed**. The goldset's labeling is what
  the reviewer must adjudicate.
- 2 cases have alias-normalised gold IDs that exist in the KB but
  were excluded for `allowed_use` reasons that the filter correctly
  enforces.

In no case is the retriever at fault. Candidate recall@50 across
BM25, dense, and hybrid is 0.9865.

## Governance filter must not be weakened to satisfy internal recall

This document and the adjudication module enforce a hard rule:
**reviewer decisions cannot change the source-tier filter, the
allowed_use vocabulary, or any live-agent retrieval behaviour**. The
only outputs of adjudication are:

- proposed edits to per-case `expected_source_ids` in the goldset
  (under the no-read protocol),
- moves between the patient-facing and clinician-facing goldsets
  (when a clinician-facing goldset exists; see
  `Data/evals/rag/clinician_facing_retrieval_goldset.README.md`),
- documented `keep_expected_sources` decisions when the goldset
  intentionally encodes the governance tradeoff.

## How to complete the packet

1. The reviewer reads only the protocol doc
   (`docs/evals/no_read_rag_goldset_protocol.md`) plus this file.
2. They open
   `Data/evals/rag/source_filter_drop_adjudication_packet.json`.
3. For each item:
   - Pick one of `adjudication_options` and assign it to
     `reviewer_decision`.
   - Fill `reviewer_role` with a role descriptor
     (`external_peer_engineer`, `external_clinician`, etc.).
   - Fill `reviewer_notes` with a one-paragraph rationale. Required
     when the decision is `revise_patient_facing_expected_sources`
     or `split_patient_and_clinician_cases`.
   - Required when the decision is `move_to_clinician_facing_goldset`
     or `split_patient_and_clinician_cases`: set `reviewer_role`.
4. They run:
   ```
   python scripts/validate_rag_goldset_adjudication.py
   ```
   The validator refuses to accept a filled packet with missing
   notes/role or with an out-of-vocabulary decision.
5. They commit the filled packet with a tag like
   `goldset-adjudication: <reviewer_role>`.
6. A separate PR — gated by reviewer attestation under the no-read
   protocol — applies the adjudicated changes to the goldset. The
   adjudication validator never auto-applies anything.

## Decisions that REQUIRE notes / role

| Decision | reviewer_notes required | reviewer_role required |
|---|:---:|:---:|
| keep_expected_sources | no | no |
| revise_patient_facing_expected_sources | **yes** | no |
| move_to_clinician_facing_goldset | no | **yes** |
| split_patient_and_clinician_cases | **yes** | **yes** |
| mark_ambiguous_needs_external_review | no | no |

The validator's source-of-truth is
`backend.services.rag_goldset_adjudication.DECISIONS_REQUIRING_NOTES`
and `DECISIONS_REQUIRING_REVIEWER_ROLE`.

## Files

- Module: [`backend/services/rag_goldset_adjudication.py`](../../backend/services/rag_goldset_adjudication.py)
- Packet builder: [`scripts/run_rag_goldset_adjudication_packet.py`](../../scripts/run_rag_goldset_adjudication_packet.py)
- Validator: [`scripts/validate_rag_goldset_adjudication.py`](../../scripts/validate_rag_goldset_adjudication.py)
- Packet artifact: `Data/evals/rag/source_filter_drop_adjudication_packet.json`
- Readiness artifact: `Data/evals/rag/latest_goldset_adjudication_readiness.json`
- Clinician-facing goldset placeholder:
  `Data/evals/rag/clinician_facing_retrieval_goldset.README.md`
- Related ADRs:
  [ADR 0005 (held-out variants)](../adr/0005-adversarial-holdout-variants.md),
  [ADR 0009 (source alias normalisation)](../adr/0009-source-alias-normalization.md).

## What this is NOT

- **Not clinical validation.** No clinician sign-off, no IRB, no real
  patient data.
- **Not retrieval improvement.** No ranking, governance, or live-agent
  behaviour is changed by adjudication.
- **Not goldset mutation.** The packet records a SHA-256 of the
  goldset at packet-build time; any later mismatch trips the
  validator's `packet_did_not_mutate_goldset` check.
- **Not auto-apply.** Filled decisions still need a separate
  reviewer-attested PR before they affect any artifact.
