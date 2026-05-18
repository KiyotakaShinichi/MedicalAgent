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
- Medical advisor readiness packet

## Accepted `needs_attention`

Some artifacts are allowed to be `needs_attention` because their job is to
surface known weaknesses, not hide them:

- RAG tier ablation
- shortcut audit
- toxicity feature audit

Those statuses must remain visible in the release output and reviewer docs.
They are not clinical validation.

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
