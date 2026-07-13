# Longitudinal context cards

> **Deterministic, auditable, provenance-stamped reference cards.** No
> open-ended memory, no chain-of-thought, no clinical claim. Not
> clinical validation.

Each patient gets 7 cards:

1. **latest_cbc_trend** — last 3 cycles' nadir WBC / Hgb / Plt.
2. **symptom_trend** — last 3 cycles' max severity + count.
3. **imaging_summary_trend** — last 3 cycles' MRI tumor size + % change.
4. **medication_treatment_context** — most recent regimen + cycle + dose flags.
5. **missing_evidence** — explicit list of required fields absent in the timeline.
6. **review_flags** — urgent / support / toxicity / severe-symptom flags from the most recent row.
7. **last_safety_escalation** — most recent cycle where `urgent_intervention_needed == 1`.

Every card carries:

- `provenance`: the source CSV path + the exact row indices used.
- `timestamps`: the source row dates.
- `missing_evidence`: explicit list of fields the card cannot fill.
- `card_disclaimer`: "Synthetic engineering signal · Not a clinical
  prediction · For clinician review" (always present).
- `clinical_validation: false`.

## Files

- Module: [`backend/services/longitudinal_context_cards.py`](../backend/services/longitudinal_context_cards.py)
- Script: [`scripts/run_longitudinal_context_cards.py`](../scripts/run_longitudinal_context_cards.py)
- Artifact: [`Data/evals/ops/latest_longitudinal_context_card_eval.json`](../Data/evals/ops/latest_longitudinal_context_card_eval.json)
- Tests: [`tests/test_frontier_engineering_layers.py`](../tests/test_frontier_engineering_layers.py)

## Current honest result (50-patient sample, 350 cards)

| Metric | Value |
|---|---:|
| provenance_coverage | 1.0 |
| timestamp_coverage | 1.0 |
| missing_evidence_disclosure_rate | 0.0 |
| unsafe_inference_rate | 0.0 |

**Honest reading**: every card is provenance-stamped and
timestamp-bearing. `missing_evidence_disclosure_rate = 0.0` is the
honest report — the synthetic data has every required field filled,
so there is nothing to disclose. On a real cohort this number would
move; on this synthetic timeline it stays at zero by construction.

`unsafe_inference_rate = 0.0` confirms no card summary contains any
of the forbidden inference tokens (`you have cancer`, `you should
stop`, etc.).

## What these cards are NOT

- Not chain-of-thought storage. The `to_dict` shape is closed; no
  free-form generation is captured.
- Not a clinical signal. The disclaimer is per-card, not just
  per-artifact.
- Not patient-facing without a clinician review boundary. A
  downstream UI would still wrap the card in the existing slim
  clinical-boundary banner.

## Related

- [`docs/per_turn_trace.md`](per_turn_trace.md) — for the no-CoT contract.
- [`docs/synthetic_data_quality.md`](synthetic_data_quality.md)
