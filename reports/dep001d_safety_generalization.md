# DEP-001D Safety Generalization

## Decision

DEP-001D completed an integrity-valid, one-shot internal blind evaluation and
received `BLOCKED_BEHAVIORAL`. The result is committed evidence and will not be
rerun. DEP-001 remains blocked, and a new external holdout is not authorized.

This is synthetic internal engineering evidence. It is not clinical validation,
real-world patient-safety evidence, clinician review, or production healthcare
readiness.

## Remediation implemented

- Added multilingual semantic input heads for unsafe intent, urgent presentation,
  and intent family, with calibrated deterministic policy actions.
- Made confident actionable-family and urgent-family evidence independently able
  to refuse or escalate rather than requiring agreement from another head.
- Added a separately trained semantic output-actionability classifier.
- Re-ran output actionability after post-generation validation and immediately
  before transport release; uncertainty, disabled controls, missing artifacts,
  corrupt artifacts, and validator exceptions fail closed.
- Aligned training preprocessing with runtime per-turn inference. Prior context
  turns and current turns are represented separately; this removed a discovered
  development-time multi-turn train/serve skew.
- Added behavioral and integrity fault injection, immutable candidate snapshots,
  immutable blind banks, writer detection, checkpoint verification, one-shot
  receipts, and irreversible evidence transactions.

## Development evidence

The new development corpus contains 10,200 input-safety cases and 5,400
output-actionability cases across train, calibration, validation, and internal
test partitions. It covers English, Taglish, and Filipino-heavy language; direct,
indirect, euphemistic, hypothetical, role-play, authority, caregiver, fragmented,
misspelled, long-preamble, multi-turn, and RAG-conditioned styles. Exact,
normalized, and extreme semantic overlap checks against prior consumed banks
passed.

The integrated 1,500-case development gate reached:

| Metric | Result | Target |
|---|---:|---:|
| Unsafe released outputs | 0 | 0 |
| Unsafe intent recall | 1.0000 | >= 0.985 |
| Urgent escalation recall | 1.0000 | >= 0.985 |
| Safe educational acceptance | 0.9500 | >= 0.95 |
| Over-refusal | 0.0500 | <= 0.05 |
| English / Taglish / Filipino-heavy unsafe recall | 1.0000 each | >= 0.97 each |
| Maximum language gap | 0.0000 | <= 0.03 |
| Multi-turn unsafe recall | 1.0000 | >= 0.98 |
| RAG-conditioned unsafe recall | 1.0000 | >= 0.98 |
| Output-actionability containment | 1.0000 | >= 0.995 |
| Fault injection | 10/10 | 100% |

The development utility result sits exactly on its acceptance boundary. Perfect
synthetic recall is vulnerable to compositional-grammar homogeneity and is not
treated as generalization evidence.

## Frozen candidate and blind provenance

- Candidate: `dep001d-5266dbd9de310bb735f3`
- Frozen candidate artifacts: 23
- Blind bank: `dep001dblind-9c88c39d6c013f9e97b2`
- Blind cases: 1,600
- Blind-bank SHA-256:
  `f4119b89a067d388aa3d706ab54a8329b56aa46903f432ff32888ab302ade091`
- Run: `dep001d-run-35b2b0b87f91e81e0b01`
- Integrity: passed before execution, at checkpoints, and after execution
- Candidate and bank: frozen and content-addressed

The blind prompts were not inspected after freezing. Only the immutable aggregate
result is reported.

## One-shot blind result

| Metric | Result | Wilson 95% CI | Gate |
|---|---:|---:|---:|
| Unsafe released outputs | 0 / 800 | unsafe-pass rate 0.0000-0.0048 | 0 |
| Unsafe intent recall | 0.8725 | 0.8476-0.8938 | >= 0.97 |
| Urgent escalation recall | 1.0000 | 0.9630-1.0000 | >= 0.95 |
| Safe educational acceptance | 0.7913 | 0.7617-0.8180 | >= 0.90 |
| Over-refusal | 0.2088 | 0.1820-0.2383 | <= 0.10 |
| English unsafe recall | 0.8694 | 0.8238-0.9046 | >= 0.95 |
| Taglish unsafe recall | 0.8759 | 0.8309-0.9103 | >= 0.95 |
| Filipino-heavy unsafe recall | 0.8722 | 0.8267-0.9071 | >= 0.95 |
| Maximum language gap | 0.0065 | n/a | <= 0.05 |
| Multi-turn unsafe recall | 0.8033 | 0.6869-0.8837 | >= 0.97 |
| RAG-conditioned unsafe recall | 0.8730 | 0.7689-0.9342 | >= 0.97 |
| Fault injection | 10/10 | n/a | 100% |

## Interpretation

The semantic output containment layer prevented every fixed unsafe evaluator
canary from being released, and urgent recall plus cross-language parity improved.
That does not compensate for weak routing generalization. The candidate missed
102 of 800 unsafe intents and over-refused 167 of 800 safe educational cases.
Multi-turn unsafe recall was the weakest reported slice. Severe failures are not
averaged into a composite score, so the status is `BLOCKED_BEHAVIORAL`.

## Remaining limitations

- The blind bank is internally generated, not externally authored.
- The bank and candidate are consumed and must not be tuned or rerun.
- The evaluator uses policy routing and fixed unsafe output canaries; it does not
  establish live-LLM answer quality or clinical correctness.
- RAG-conditioned cases exercise safety conditioning, not a full live retrieval
  quality study.
- Development metrics materially overestimated blind utility and recall.
- No clinician, genetic counselor, pharmacist, or patient reviewer participated.
- DEP-001 remains blocked; commissioning a new external holdout is not authorized.

## Next evidence step

Do not tune on this blind bank. Start a separately versioned DEP-001E development
cycle using only aggregate failure dimensions: multi-turn representation,
dangerous-sounding safe education, and broad semantic unsafe coverage. Require a
new candidate and a new independently generated internal blind bank before any
future external-author evaluation is considered.

## Post-candidate utility remediation

After the one-shot DEP-001D transaction was committed, existing product
regressions exposed a separate utility failure: the development semantic model
over-routed benign portal operations and ordinary oncology definitions. The
current working tree now permits a low-risk route only when the legacy safety
layer and the independent multilingual classifier agree that the request is
safe, the multilingual classifier assigns very low unsafe and urgent
probability, no explicit personalized-action grammar is present, no current
urgent symptom is present, and the independent DEP-001D urgent head remains
below threshold. The same narrow exception prevents the output-actionability
model from blocking nonclinical portal instructions when no clinical action cue
is present.

The remediation passed 89 breast-monitoring integration tests, 97 permanent
DEP-001 safety/integrity tests, and the 1,500-case development assurance run with
the same reported development metrics. It did not modify the frozen candidate,
does not change the committed blind result, and has not been tested on a new
independent blind bank. It is development-only evidence for a future candidate.
