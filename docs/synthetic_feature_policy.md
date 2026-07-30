# Synthetic feature policy

NLCare keeps old serialized synthetic models readable, but it does not treat
their legacy feature set as promotion evidence. The legacy set includes
`mri_percent_change_from_baseline`, which is definitionally close to the
synthetic regression target `response_score_percent`.

All new synthetic promotion, retraining, and perturbation comparisons use
`synthetic_proxy_removed_promotion_v1`. This policy removes the direct response
proxy and requires candidates to pass the existing leakage, shortcut,
calibration, subgroup, perturbation, and cross-generator checks.

This is an evidence-hygiene correction, not a clinical improvement. The data
and labels remain simulator-built. A model passing this policy is still
monitor-only engineering evidence and is not a diagnostic, treatment,
prognostic, or clinically validated model.
