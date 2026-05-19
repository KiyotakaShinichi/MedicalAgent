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

## Accepted `needs_attention`

Some artifacts are allowed to be `needs_attention` because their job is to
surface known weaknesses, not hide them:

- RAG tier ablation
- shortcut audit
- toxicity feature audit

Those statuses must remain visible in the release output and reviewer docs.
They are not clinical validation.

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
