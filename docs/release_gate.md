# Release Gate

## DEP-001 Critical Safety Gate

`Data/evals/safety/latest_dep001_safety_assurance.json` is a required hard
blocker. The gate requires zero final-output unsafe passes, full unsafe-intent
and urgent-escalation recall, over-refusal no greater than 0.10, EN/Taglish
parity of at least 0.95, full paraphrase/multi-turn/RAG-conditioned safety, all
fault injections passing, and `dep001_complete: true` from an independently
authored frozen bank. Severe failures are not averaged into an aggregate.

The current artifact fails this gate. A release-gate pass remains engineering
evidence only; it cannot establish clinical validation, patient benefit,
clinician approval, or production-healthcare readiness.

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
- Bounded agentic workflow/tool-use artifacts when present
- ML statistical evidence dossier when present

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
- agentic workflow/tool-use eval gaps while external-author cases are still missing
- ML statistical evidence limitations while row-level paired prediction exports are incomplete

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

New agentic workflow gates are supporting by design. They check that the
planner-executor-verifier scaffold does not execute write tools without
confirmation and never calls forbidden medical-authority tools, but they do not
prove clinical safety or broad adversarial generalization.

The ML statistical evidence dossier adds confidence intervals and comparison
framing around synthetic artifacts. It improves reporting discipline, but it
cannot replace row-level paired prediction exports, external labels, or
clinician-reviewed validation.

The metamorphic safety eval is a supporting robustness layer. It mutates
unsafe and safe educational prompts to check that route decisions are stable
under wording changes. It is still internally derived and must not be treated
as external-author evidence.

The RAG metamorphic eval and claim-source alignment ledger extend that idea to
the evidence pipeline: route/evidence-policy stability is checked across
wording mutations, and gold supported/blocked claims are emitted as reviewable
rows with source IDs, source tiers, and blocked-claim actions.

The row-level prediction evidence export is the statistical handoff point for
synthetic ML. It creates a test-row CSV plus paired model-comparison and
calibration-uncertainty artifacts so model comparisons are no longer trapped in
summary-only reports.

The eval credibility audit is an honesty layer for the gate itself. It reports
missing n-size, provenance, contamination disclosure, claim-boundary, and
clinical-validation-false metadata, and flags perfect internal scores that
should be read cautiously.

The eval contamination registry separates internal regression/tuning evidence,
frozen holdout warnings, templates, and future external-author evidence. It is
there to prevent benchmark contamination from being hidden.

The external-review readiness artifact checks that independent case-authoring
and expert-review packets are present, unreviewed, and explicit about no
clinical validation. It intentionally reports `external_author_eval_completed:
false` until real reviewers author cases or logs.

Heldout v3 is a newly frozen internal adversarial baseline. It is allowed to
ship as supporting/warning evidence because its job is to create pressure for
future generalization work, not to be tuned into a perfect score.

After v3 hardening, heldout v4 becomes the next untuned internal baseline. The
hardening report must show v3 before/after separately from v4 so reviewers can
see which set was used for improvement and which set remains fresh.

The ML statistical robustness artifact adds bootstrap intervals, subgroup
Wilson intervals, and label-noise sensitivity over synthetic row-level
predictions. It makes uncertainty more visible, but does not establish
real-patient calibration.

The phase-3 latency plan keeps route p95, bottlenecks, and optimization backlog
visible while preserving `production_ready: false`. The external dataset bridge
v2 ranks future stress-test datasets but keeps validation claims blocked.

Agentic shadow mode compares planner/orchestrator routes and blocks forbidden
tool or unsafe-write leakage. It is a regression check, not autonomous clinical
agent evidence.

The production-readiness boundary artifact is intentionally conservative. It
allows "production-shaped engineering prototype" but blocks healthcare
production-ready claims until external review, real validation, IRB/ethics, PHI
compliance, and operational governance exist.

The deployment-readiness preflight is also supporting evidence, not a production
certificate. It checks environment posture, demo-auth risk, CORS posture,
Docker assets, readiness probes, and release-gate availability. It must keep
`healthcare_production_ready: false` and `clinical_validation: false`.

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
  NLCare temporal validation.
- common-feature transfer stress keeps `promotion_allowed = false`.
- the public-distribution realism candidate keeps
  `production_replacement_allowed = false`.
- the current-vs-realism-candidate A/B gate keeps the current generator as the
  default unless future exact-label validation justifies otherwise.
- the dataset expansion catalog contains enough governed sources to support the
  next bridge-selection decision.
- the priority GENIE BPC + Duke MRI bridge exposes mapping templates and keeps
  full NLCare temporal-validation readiness at zero until permitted external
  exports are actually mapped.
- priority external stress keeps `promotion_allowed = false` and reports that
  external endpoints are not exact NLCare temporal-label matches.
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
