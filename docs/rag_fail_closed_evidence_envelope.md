# Fail-Closed RAG Evidence Envelope

## Scope

NLCare now applies a programmatic release invariant to patient-visible agent
responses:

> No valid evidence envelope means no evidence-dependent medical answer.

This is engineering hardening for a synthetic, non-diagnostic prototype. It
reduces unsupported-answer release when retrieval or validation is incomplete;
it does not guarantee factual correctness, clinical safety, or clinical
validation.

## Response Flow

The patient agent now follows this release sequence:

1. Classify safety and intent.
2. Check an exact or semantic cache only if the route is cache-eligible.
3. Retrieve, source-filter, rerank, and compress evidence when RAG is required.
4. Generate a candidate and validate its basic answer/citation schema.
5. Run output and post-generation medical-boundary checks.
6. Select the intent-specific source policy, validate claim support, grade
   evidence, and classify retrieval uncertainty.
7. Build and authorize the evidence envelope.
8. Persist traces/evaluation logs and write a cache entry only after an
   `ALLOW` decision.
9. Revalidate the response digest and envelope at the JSON or SSE transport
   boundary.

The support-chat wrapper performs the same final check after emotional wording,
record provenance, tool confirmations, or alert notices have been assembled.
An evidence-dependent reply changed after authorization is rejected rather than
silently re-authorized against stale claim mappings.

## Typed Contract

`backend/services/rag_evidence_envelope.py` defines the immutable
`EvidenceEnvelope` and `AuthorizationDecision` types. The envelope records:

- request ID and versioned release, safety, and validator policies;
- whether evidence is required and the retrieval/answerability states;
- chunk/source IDs, source tier, allowed use, staleness status, and references;
- hashed claims and claim-to-source chunk mappings;
- citation, claim-support, evidence-coverage, conflict, and safety states;
- validation error/warning codes, abstention reason, response digests, and
  trace metadata.

Raw queries and retrieved passages are not copied into envelope events. A
query hash and source/chunk references are used for PHI-minimized correlation.

## Closed Dispositions

Only the exact `ALLOW` enum value can release evidence-dependent content:

- `ALLOW`
- `ABSTAIN_INSUFFICIENT_EVIDENCE`
- `ABSTAIN_VALIDATION_FAILURE`
- `ABSTAIN_CONFLICTING_EVIDENCE`
- `ABSTAIN_UNSUPPORTED_CLAIMS`
- `BLOCK_MEDICAL_BOUNDARY`
- `BLOCK_SAFETY`
- `INTERNAL_ERROR`

Missing fields, unknown statuses, malformed values, unsupported versions, and
exceptions deny release. Citations, retrieval confidence, model confidence, or
a disclaimer cannot authorize an answer by themselves.

## Failure Behavior

The final policy requires all applicable checks to be complete. It abstains
when retrieval is empty, source metadata is incomplete, answerability is
unknown, claim/citation coverage is partial, evidence conflicts, or any
component reports an error. The intent-aware RAG layer also immediately clears
the candidate and citations if mode selection, source filtering, claim
validation, grading, or uncertainty classification raises.

Safe English and Taglish responses distinguish insufficient evidence,
conflicting evidence, unsupported claims, medical-boundary blocks, and internal
validation failures. Emergency language is preserved only when the input
independently triggered urgent-safety policy.

## Cache Contract

Cache schema `agent_response_cache_v5` stores the authorized envelope and the
RAG governance metadata needed to reconstruct the response. Reads require:

- a parseable current envelope;
- exact envelope, release-policy, safety-policy, and validator versions;
- a matching knowledge-base fingerprint and unexpired row;
- a response digest matching the authorized payload;
- a current `ALLOW` decision.

Legacy or incompatible rows are rejected and naturally repopulated after a
fresh authorized request. There is no permissive migration that treats old
rows as valid.

## Streaming Contract

SSE uses buffered post-guardrail display streaming. It may emit progress and a
PHI-free `blocked_pending_validation` event while work is in progress, but it
does not emit `answer_delta` content until the complete reply has passed the
transport authorization check. A missing or mutated envelope becomes a safe
abstention before the first answer token.

## Operational Signals

PHI-safe response-local events cover validation start/completion, envelope
creation, validator failure, release allow/deny, abstention reason, cache
rejection, version mismatch, and streaming authorization state. In-process
counters include:

- `rag_release_allowed_total`
- `rag_release_denied_total`
- `rag_abstention_total`
- `rag_validation_failure_total`
- `rag_unsupported_claim_total`
- `rag_cache_rejected_total`
- `rag_envelope_version_mismatch_total`

These counters are prototype observability, not a durable multi-process metrics
backend.

## Fault Coverage

`tests/test_rag_fail_closed_evidence_envelope.py` covers more than 30
deterministic faults and invariants, including retrieval/generation/validator
failures, all five intent-aware governance dependencies, incomplete claim
support, conflicts, unknown/malformed envelopes, policy-version cache
invalidation, post-authorization mutation, alternate nested payloads, English
and Taglish behavior, PHI-safe events, and buffered SSE. A positive case proves
that complete evidence, claim mappings, citations, safety checks, and the exact
`ALLOW` disposition still release a legitimate answer.

## Known Limitations

- The default claim validator still relies on lexical heuristics unless the
  optional semantic validator is available. Fail-closed availability does not
  make that validator medical-grade entailment.
- Source quality and metadata can still be wrong at ingestion time.
- In-process counters are not durable across processes or restarts.
- Internal tests and synthetic data are engineering evidence only. No external
  clinician review, real-patient validation, or production-healthcare readiness
  is established.

## Verification

```powershell
python -m pytest tests/test_rag_fail_closed_evidence_envelope.py -q
python -m pytest tests/test_post_gen_answer_tier_escalation.py tests/test_rag_claim_structured_alignment.py -q
python -m pytest tests/test_breast_monitoring.py -q
python scripts/ship.py
```
