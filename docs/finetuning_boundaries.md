# OncoTrack fine-tuning boundaries

OncoTrack's fine-tuning scaffold trains behavior / style, not medical
knowledge. This document is the hard-rule reference; if a future
contributor proposes adding training examples that violate one of
these rules, the dataset must reject them.

## Allowed behavior targets
The dataset preparer's allow-list (`scripts/prepare_finetune_dataset.py`)
accepts examples in **five** behavior categories:

1. `clinician_summary`            — structured chart-handoff summaries
2. `missing_data_disclosure`      — "I don't have enough detail; please share …" phrasing
3. `questions_to_ask_care_team`   — preparing patient questions for visits
4. `supplement_boundary`          — non-definitive safety routing for supplements
5. `taglish_safety`               — Filipino/Taglish patient-safe refusal phrasing

## Blocked claims
The dataset preparer's safety filter rejects any example whose
assistant string trips:

- `diagnosis`
- `treatment_recommendation`
- `dosage_change`
- `prognosis_estimate`
- `genetic_risk_prediction`
- `tumor_marker_conclusion`
- `survival_estimate`
- `supplement_safe_with_chemo_claim`
- `replace_treatment_claim`
- any phrase the `medical_claim_boundary.classify_medical_claim`
  service marks as unsafe

A rejected example is logged in the dataset card's
`rejected_examples` list with its violation list, and is NOT included
in the prepared dataset.

## Hard rules
1. **No real patient data.** All examples are synthetic.
2. **No clinical knowledge tuning.** The base model already has the
   factual layer it has; fine-tuning here changes tone, format, and
   refusal style, not medical facts.
3. **Dry-run by default.** `scripts/run_lora_finetune_dryrun.py`
   produces a manifest + model-card stub without loading weights.
   Real training is gated on GPU availability + a future contributor's
   explicit decision; the dataset card and the model card must
   document it.
4. **Post-deployment safety layers still apply.** Any tuned adapter
   that ships must still pass:
   - `agent_safety.safety_scope_check`
   - `agent_input_gate.input_guardrail_check`
   - `agent_output_gate.output_guardrail_check`
   - `agent_post_gen.apply_post_gen_validator`
   - `medical_claim_boundary.classify_medical_claim`
   - `rag_claim_validator.validate_claims`
5. **A/B vetting before ship.** A tuned adapter must beat the baseline
   in the offline A/B framework (`scripts/run_offline_ab_eval.py`)
   with NO safety regression (see `docs/offline_ab_testing.md`).
6. **Clinician review before ship.** Even a passing A/B is engineering
   evidence only — clinician review of the prepared dataset and the
   tuned adapter's outputs is required before any real-patient
   exposure.

## Dataset card fields (required)
Every prepared dataset emits a JSON dataset card with:

- `dataset_purpose`
- `allowed_behavior_targets`
- `blocked_claims`
- `synthetic_or_source`
- `safety_filters_applied`
- `known_risks`
- `example_counts.{accepted_total, rejected_total, by_behavior}`
- `system_prompt`
- `rejected_examples`
- `files.{dataset_jsonl, dataset_card}`
- `claim_boundary`

## Model card stub fields (required for any future adapter)
- `base_model`
- `adapter_name`
- `intended_behavior` (must be a subset of allowed behavior targets)
- `not_intended_for`
- `evaluation_results`
- `safety_validator_compatibility`
- `claim_boundary`

## How to run

```bash
# 1) prepare a dataset from the templates
python scripts/prepare_finetune_dataset.py

# 2) emit a dry-run training manifest + model-card stub (no GPU needed)
python scripts/run_lora_finetune_dryrun.py

# 3) evaluate the prepared dataset against the safety contract
python scripts/evaluate_finetuned_behavior.py
```

## Claim boundary
This scaffold does not train medical knowledge. It trains tone,
format, and refusal style on synthetic examples whose claims are
bounded by the medical claim boundary checker. Any deployment of a
tuned adapter remains a **non-diagnostic clinician-in-the-loop
engineering prototype**.
