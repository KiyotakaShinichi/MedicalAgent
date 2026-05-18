# MLE defensibility (synthetic-only)

This page documents the **engineering audits** the MLE stack ships
today. Every metric here is computed on **synthetic data**. None of
these audits establish clinical validity.

## What's in place

| Audit | Module | Artifact |
|---|---|---|
| Shortcut audit | `backend/services/shortcut_audit.py` | `Data/evals/models/latest_shortcut_audit.json` |
| Toxicity feature audit | `backend/services/toxicity_feature_audit.py` | `Data/evals/models/latest_toxicity_feature_audit.json` |
| Counterfactual stability | `backend/services/counterfactual_stability.py` | `Data/evals/models/latest_counterfactual_stability.json` |
| Per-head calibration | `backend/services/per_head_calibration.py` | `Data/evals/models/latest_per_head_calibration.json` |
| Modality robustness comparison | (existing) | `Data/evals/models/latest_modality_robustness_comparison.json` |
| Response conformal calibration | (existing) | `Data/evals/models/latest_response_conformal_calibration.json` |
| Hybrid subgroup metrics | `backend/services/hybrid_subgroup_metrics.py` | `Data/evals/models/latest_hybrid_subgroup_metrics.json` |
| Synthetic realism hardening | `backend/services/synthetic_realism_hardening.py` | `Data/evals/models/latest_synthetic_realism_hardening.json` |
| Soft toxicity target benchmark | `backend/services/soft_toxicity_target_benchmark.py` | `Data/evals/models/latest_soft_toxicity_target_benchmark.json` |
| Learned abstention (experimental) | `backend/services/learned_abstention.py` + `_experiment.py` | `Data/evals/models/latest_learned_abstention.json` |
| Synthetic generator card | (existing) | `Data/evals/models/latest_synthetic_generator_card.json` |
| Self-supervised timeline | `backend/services/self_supervised_timeline.py` | `Data/evals/models/latest_self_supervised_timeline.json` |

## Promotion gate (PART 2.6 — new this cycle)

`scripts/run_mle_promotion_gate.py` aggregates every audit above and
returns one of three decisions:

- **PROMOTE** — every condition passes
- **HOLD** — non-critical failures only, or critical failures with
  `decision_on_fail: "HOLD"`
- **REJECT** — at least one critical condition with
  `decision_on_fail: "REJECT"` failed

Thresholds live in `config/mle_promotion_thresholds.yaml`. The gate is
**read-only** — it reads existing artifacts, never retrains, never
mutates.

Currently the gate reports **HOLD** because
`latest_synthetic_realism_hardening.json` is at `needs_attention` —
that's honest self-reporting given the synthetic-only constraint.

## What this layer is

An aggregated engineering opinion over the synthetic stack. PROMOTE
here means "the audit proxies are within their thresholds" — useful
as a smoke check, not as a deployment decision.

## What this layer is NOT

- Not clinical validation.
- Not regulatory approval.
- Not evidence of real-world patient benefit.
- Not a substitute for clinician review of model outputs.

## Suggested promotion workflow (future)

1. Regenerate audits → confirm `Data/evals/models/latest_*` files are
   fresh.
2. `python scripts/run_mle_promotion_gate.py` → PROMOTE / HOLD / REJECT
3. If PROMOTE: still requires clinician review of a sampled set of
   outputs before any real deployment.
4. If HOLD: review which audit is at `needs_attention` and decide
   whether to fix or accept the residual risk in the model card.
5. If REJECT: stop. A critical condition failed; investigate the audit
   artifact pointed to by the gate's `reasons` field.

## Claim boundary
Engineering promotion gate over synthetic-only audits. PROMOTE is not
clinical validation, regulatory approval, or evidence of real-world
patient benefit. Production behavior remains safety-gated, source-
governed, validator-checked, and clinician-reviewed.
