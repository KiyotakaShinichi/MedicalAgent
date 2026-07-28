# XAI engineering evidence

NLCare exposes synthetic model explanations as engineering diagnostics, not
clinical explanations. The explanation stack separates four questions:

1. **Arithmetic fidelity**: does the SHAP base value plus all exported
   contributions reconstruct the model log-odds output?
2. **Presentation safety**: are mutually exclusive one-hot features grouped,
   and are near-outcome proxies separated from ordinary context?
3. **Bootstrap ranking sensitivity**: do global mean-absolute-SHAP rankings
   remain similar when synthetic patient explanations are resampled?
4. **Retraining sensitivity**: do global and local grouped explanations remain
   similar across independent patient-level splits and model fits?
5. **Comprehension contract**: does patient-facing wording include meaning,
   calculation, missingness, limitations, and a non-authoritative next step?

Artifacts:

- `Data/evals/models/latest_xai_fidelity_audit.json`
- `Data/evals/models/latest_xai_rank_stability.json`
- `Data/evals/models/latest_xai_retraining_stability.json`
- `Data/evals/models/latest_xai_comprehension_contract_eval.json`
- `Data/evals/models/latest_counterfactual_stability.json`

The bootstrap audit does not retrain the model. The separate 12-seed audit
does, and it currently reports `needs_attention`: global top-8 overlap has
p05 Jaccard `0.6`, while exact order is unstable (p05 rank correlation
`-1.0`). Local top-8 overlap is more consistent (median Jaccard `0.7778`), and
patient mean probability varies with p95 standard deviation about `0.0519`.
These are synthetic sensitivity measurements, not a target to tune away.

Neither audit proves that a feature is causal, that a person understands the
display, or that any relationship transfers to real patients. Raw
contributions belong in engineering traces; grouped factors with explicit
synthetic-only wording are the safer patient-portal surface.

Next XAI work:

- investigate why exact global rank ordering changes across retrains without
  optimizing against this one artifact;
- compare rank stability before and after shortcut-prone features are removed;
- run a structured non-clinical comprehension study before making usability
  claims;
- retain abstention and missing-modality explanations when a model head lacks
  sufficient evidence.
