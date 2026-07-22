# Governed QLoRA Behavior Experiment

This directory is an optional, synthetic-only behavior-formatting experiment.
It does not add medical knowledge and is not evidence of clinical performance.

## Current State

- No adapter has been promoted.
- Training is disabled unless an operator explicitly passes `--execute`.
- The internal frozen holdout is never used for training or checkpoint selection.
- Any successful training run remains `HOLD` until complete baseline and candidate
  generations pass the promotion gate.

## Governed Data Flow

The experiment does not own a second example bank. Export the canonical,
hash-verified training split:

```powershell
python scripts/build_qlora_behavior_dataset.py
```

The source splits and their hashes are recorded in
`Data/finetune/prepared/split_manifest.json`.

## Preflight Only

The default command verifies split hashes and reports missing execution controls.
It does not load a model or train an adapter:

```powershell
python experiments/qlora_behavior/phi3_qlora_colab.py
```

An experimental run additionally requires pinned model and tokenizer revisions,
an acknowledged license review, and a CUDA environment:

```powershell
python experiments/qlora_behavior/phi3_qlora_colab.py `
  --base-revision <immutable-revision> `
  --tokenizer-revision <immutable-revision> `
  --license-reviewed `
  --execute
```

Dependency compatibility and the model license must be reviewed before execution.
Completing this command does not authorize patient-facing use.

## Evaluation and Promotion

Generate one output per frozen case for both the baseline and candidate, then run:

```powershell
python scripts/run_finetune_promotion_gate.py
```

Hard rejection conditions include unsafe leakage, claim-boundary violations,
validator failures, incomplete generations, safety-metric regression, or any
per-behavior regression. Fewer than 50 paired cases remains `HOLD`, and meeting
the lift threshold is not described as statistical proof. `PROMOTE`, when
available, means offline shadow evaluation only.

## Claim Boundary

This experiment may target formatting, missing-data disclosure, safe refusal
language, Taglish boundary wording, and care-team question generation. It must
not target diagnosis, prognosis, treatment or dosage decisions, genetic-risk or
tumor-marker conclusions, supplement safety, or replacement of professional
review. RAG and deterministic safety controls remain outside the adapter.
