# Release Gate

`make ship` and `python scripts/ship.py` both end with:

```bash
python scripts/run_release_gate.py
```

The release gate reads `config/release_gate_thresholds.yaml` and fails if any
critical artifact is missing, stale, below threshold, or at an unaccepted
status.

## What It Checks

- Safety benchmark artifacts
- Adversarial and multilingual refusal artifacts
- Taglish safety parity
- Medical claim-boundary evaluation
- RAG benchmark, intent-aware eval, claim validation, tier ablation, and source governance
- Leakage audit
- Shortcut and toxicity feature audits
- Counterfactual stability
- Modality robustness
- Response conformal calibration
- Hybrid subgroup metrics when present
- cBioPortal clinical export and external distribution alignment when present
- Common-feature transfer stress and public-distribution realism candidate artifacts when present
- Current-vs-realism-candidate A/B gate when present
- Dataset expansion deep-search catalog when present
- Medical advisor readiness packet
- Gold claim-grounding manifest and contamination disclosure
- Cross-encoder reranker/retrieval ablation metrics
- Retrieval goldset metrics with distractors and source-tier governance checks
- Retrieval failure analysis separating true retrieval misses from metadata/goldset alias issues
- Adversarial generalization report covering original, held-out, paraphrase, and safe-negative sets
- Held-out adversarial v1 failure analysis
- Frozen internal adversarial holdout v2 generalization eval
- Unsafe-intent semantic classifier eval
- Route latency budget report
- Local latency profile by route/stage
- Phase-2 latency profile with cold-start warm-up separated from steady route timing
- Medical semantic chunking quality
- Runtime quality sentinel
- Dependency/security scan artifact when present
- Local load-test smoke artifact when present

## Accepted `needs_attention`

Some artifacts are allowed to be `needs_attention` because their job is to
surface known weaknesses, not hide them:

- RAG tier ablation
- shortcut audit
- toxicity feature audit
- held-out adversarial generalization while the gap remains visible
- frozen holdout v2 weakness while it remains explicitly reported
- reranker/retrieval goldset results when improvement is not proven
- route latency budget warnings from local-only tests

Those statuses must remain visible in the release output and reviewer docs.
They are not clinical validation.

## Hard Blockers vs Supporting Needs-Attention

Hard blockers include unsafe answers, patient-facing clinical overclaims,
leakage failures, stale or missing critical artifacts, claim-boundary
regressions, and failing breast-monitoring integration tests.

Supporting artifacts may honestly remain `needs_attention` when their role is
to document an unresolved limitation, such as strict source-tier RAG coverage
tradeoffs, synthetic generator limitations, shortcut-risk documentation, or the
unreviewed clinical advisor packet. Those are not hidden passes; they are
explicit evidence that the limitation is known and visible.

New anti-overclaim gates include semantic claim validation, over-refusal
negative controls, multilingual adversarial security, live RAG failure
analysis, and the release-gate explanation artifact.

New observability gates are supporting by design: runtime quality sentinel,
dependency scan, and local load test are meant to surface warnings without
pretending this is production SRE or certified security.

The gate deliberately separates hard blockers from warning artifacts. Unsafe
leakage, clinical overclaims, leakage failures, stale critical artifacts, and
RBAC failures block shipping. Weak held-out adversarial scores, no proven
reranker lift, or high local p95 latency are warnings that must be shown to
reviewers; they are not converted into fake pass/fail perfection.

Heldout v2 is warning/supporting evidence. It may remain `needs_attention`
because its job is to reveal generalization gaps. Do not tune against v2 and
then call it held-out; create v3 or use external-author cases first.

## Optional Public-Data Bridge Checks

The public-data bridge artifacts are optional because they depend on local
public-row exports, but when present the gate checks that:

- cBioPortal rows validate against the canonical schema and do not claim full
  OncoTrack temporal validation.
- common-feature transfer stress keeps `promotion_allowed = false`.
- the public-distribution realism candidate keeps
  `production_replacement_allowed = false`.
- the current-vs-realism-candidate A/B gate keeps the current generator as the
  default unless future exact-label validation justifies otherwise.
- the dataset expansion catalog contains enough governed sources to support the
  next bridge-selection decision.
- the priority GENIE BPC + Duke MRI bridge exposes mapping templates and keeps
  full OncoTrack temporal-validation readiness at zero until permitted external
  exports are actually mapped.
- priority external stress keeps `promotion_allowed = false` and reports that
  external endpoints are not exact OncoTrack temporal-label matches.
- mutation-context mapping keeps `promotion_allowed = false` for gene features.
- dataset fit matrix keeps production training blocked while ranking next data
  sources.
- gold claim-grounding and semantic citation artifacts exist and have no hard
  failures.
- near-boundary safety eval keeps unsafe-answer rate at zero for curated risky
  phrasing.
- uncertainty, real-data readiness, and clinical-performance dossier artifacts
  keep synthetic-only/no-clinical-validation boundaries explicit.
- medical controlled docs, event taxonomy, and ops health snapshots exist for
  reviewer readiness.

These checks are intentionally conservative. They prove interoperability and
review discipline, not clinical performance.

## Run

```bash
python scripts/run_release_gate.py
python scripts/run_release_gate.py --json
python scripts/run_release_gate.py --config config/release_gate_thresholds.yaml
```

## Claim Boundary

Passing this gate means the prototype is internally consistent enough for a
demo/review. It does not mean the model works on real patients, the medical
content is clinically approved, or the product is ready for care delivery.
