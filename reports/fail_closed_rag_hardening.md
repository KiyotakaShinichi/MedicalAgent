# Fail-Closed RAG Hardening Implementation Report

## Audit Findings Addressed

- **SAF-001:** Evidence-governance exceptions fail open. Addressed directly.
- **RAG-001:** Default claim validation is lexical overlap. The release boundary
  now abstains when validation is unavailable or incomplete, but the underlying
  semantic-quality limitation remains and is not claimed as solved.

## Root Cause

`apply_intent_aware_rag_layer` caught broad exceptions after generation but only
marked evidence as missing. The candidate reply and citations remained intact.
The finalizer had no typed, deny-by-default release decision, cache entries were
written before final evidence governance completed, and the outer support-chat
layer could mutate an authorized reply. SSE buffered generation but did not
independently require a valid final envelope before emitting answer chunks.

## Previous Fail-Open Paths

1. Exceptions in mode selection, source-tier filtering, claim validation,
   evidence grading, or uncertainty classification could preserve a candidate.
2. Missing, malformed, partial, or unknown validation states had no single final
   authorization owner.
3. Exact/semantic cache hits relied on age and KB fingerprint, not a current
   envelope and policy-version decision.
4. Successful generation could be cached before the complete release boundary.
5. Emotional/provenance/alert formatting could change the response after core
   authorization.
6. Alternate support-chat and streaming routes had no independent final
   transport assertion.
7. Top-level provider or pipeline exceptions could escape instead of returning
   a typed safe result.

## New Evidence-Envelope Contract

`backend/services/rag_evidence_envelope.py` defines a frozen typed envelope with
request and policy versions, retrieval/answerability states, governed source
metadata, passage references, hashed claims, claim-to-source mappings,
citation/support/coverage/conflict/safety states, validation codes, response
digests, trace metadata, abstention reason, and a closed disposition enum.

The schema deliberately excludes raw query text and passages from its
observability events. The envelope is bound to the exact outgoing response via
SHA-256 digest.

## Authorization Logic

`authorize_evidence_release` is the centralized deny-by-default policy. Only an
exact `ALLOW` can release an evidence-dependent answer. It requires successful
retrieval, `answerable_with_citations`, governed evidence and source IDs,
complete citation and claim support, complete evidence coverage, no unresolved
conflict, no validation errors, and passed safety validation. Missing, unknown,
malformed, partial, failed, or incompatible states deny.

`enforce_transport_release` repeats the decision immediately before JSON/SSE
transport and checks that the reply digest still matches. Non-allowing payloads
must have no citations.

## Files Changed

- `backend/services/rag_evidence_envelope.py` (new)
- `backend/services/agent_post_gen.py`
- `backend/services/agent_rag.py`
- `backend/services/agent_cache.py`
- `backend/services/support_chat_agent.py`
- `backend/api/routers/patient_interactions.py`
- `backend/services/fail_closed_rag_assurance.py` (new)
- `scripts/run_fail_closed_rag_assurance.py` (new)
- `scripts/ship.py`
- `config/release_gate_thresholds.yaml`
- `docker-compose.yml`
- `docker-compose.prod.yml`
- `tests/test_rag_fail_closed_evidence_envelope.py` (new)
- `tests/test_fail_closed_rag_assurance.py` (new)
- `tests/test_deployment_worker_wiring.py` (new)
- `docs/rag_fail_closed_evidence_envelope.md` (new)
- `reports/fail_closed_rag_hardening.md` (new)

## Tests Added

The new file currently contains **68 collected tests**:

- 42 parameterized release-boundary fault cases;
- 9 malformed/unknown envelope cases;
- 5 independently faulted governance dependency stages;
- positive complete-envelope `ALLOW` coverage;
- cache policy/version and mutated-payload rejection;
- post-authorization response mutation and nested-route rejection;
- Taglish abstention parity and no false emergency escalation;
- PHI-safe event and malformed-observability-sink behavior;
- top-level agent exception and intent-aware validator exception behavior;
- buffered streaming enforcement and closed-enum invariants.

Existing chat/RAG tests are also used to verify legitimate direct support,
record confirmation, safety, and cache behavior.

## Commands Executed and Results

Commands used:

```powershell
python -m py_compile backend/services/rag_evidence_envelope.py backend/services/agent_cache.py backend/services/agent_post_gen.py backend/services/agent_rag.py backend/services/support_chat_agent.py backend/api/routers/patient_interactions.py
python -m pytest tests/test_rag_fail_closed_evidence_envelope.py -q
python -m pytest tests/test_breast_monitoring.py -k "chat_ or agent_rag" -q
python -m pytest tests/test_post_gen_answer_tier_escalation.py tests/test_rag_claim_structured_alignment.py -q
python -m pytest tests/test_breast_monitoring.py -q
python -m pytest tests/test_fail_closed_rag_assurance.py tests/test_deployment_worker_wiring.py tests/test_release_decision_surface.py tests/test_ship_runner.py -q
python scripts/run_fail_closed_rag_assurance.py
python scripts/ship.py
```

Verified results:

- compile check: passed;
- fail-closed fault suite: 68 passed;
- assurance/deployment/release-runner tests: 24 passed;
- focused post-generation and structured-claim compatibility: 77 passed;
- breast-monitoring integration suite: 80 passed;
- focused chat/RAG compatibility: one deterministic MRI-confirmation regression
  was found, corrected, and its original test now passes.
- the fresh fail-closed assurance artifact reports `status: passed`;
- the full ship workflow passed all 70 selected steps, including 80 breast-
  monitoring tests, 144 assurance/XAI/automation tests, 57 frontend unit tests,
  16 Playwright smoke tests, lint, production build, evidence refresh, and the
  release gate;
- the release gate passed with 229 artifacts, 28 decision artifacts, zero hard
  failures, and one warning for non-zero adversarial-generalization leakage;
- the repository secret scan has zero findings. A test-fixture false positive
  was removed by renaming the fixture marker without changing scanner rules.

## Security and Privacy Considerations

- Envelope events store request IDs, disposition/reason codes, timestamps, and
  hashes/references rather than raw user queries, prompts, responses, or
  evidence passages.
- Exception messages are not returned or copied into patient-facing traces;
  only exception type and bounded error code are retained.
- Cache entries without current policy/envelope versions are invalidated rather
  than interpreted as safe legacy data.
- Streaming emits no evidence-dependent answer token while authorization is
  pending.
- Observability-sink failures produce an internal-error abstention rather than
  permitting release.

## Unexecuted Evidence

Real provider reconciliation, managed cloud deployment, external notification
delivery and acknowledgement, external review, and clinical validation are not
executed by this hardening suite.

## Remaining Limitations

1. This availability/integrity boundary does not prove that a validator's
   positive judgment is medically correct.
2. The default lexical claim validator remains weaker than medical entailment
   and contradiction modeling; RAG-001 is only partially mitigated.
3. The KB and source metadata can contain ingestion or labeling errors.
4. Metrics are process-local; durable cross-instance metrics require an
   external telemetry backend.
5. Internal synthetic tests are not clinical validation, clinician approval,
   real-world safety evidence, or production-healthcare readiness.

## Recommended Next Task

Build a frozen high-risk claim/source entailment set for negation, temporality,
numeric units, population scope, treatment direction, VUS, tumor markers, and
supplements. Evaluate lexical, NLI, and deterministic contradiction layers
separately, and make unavailable semantic validation abstain for those high-risk
claim classes. This addresses the remaining RAG-001 correctness gap without
weakening the fail-closed boundary implemented here.
