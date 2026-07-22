# NLCare behavior fine-tuning scaffold

This directory contains synthetic examples for behavior and format experiments
only. It contains no real patient data and is not a medical-knowledge tuning
dataset.

## Layout

```text
templates/                              internally authored source examples
prepared/dataset.jsonl                  accepted examples with split labels
prepared/dataset_train.jsonl            only split permitted for training
prepared/dataset_development.jsonl      iteration and early-stopping split
prepared/dataset_internal_frozen_holdout.jsonl
prepared/dataset_card.json              provenance, risks, rejections, counts
prepared/split_manifest.json            source/split hashes and contamination
runs/latest_dryrun_manifest.json         no-training execution plan
runs/latest_model_card.json              adapter card template
evaluations/                             future baseline/candidate generations
```

The internal frozen holdout was created from internally authored templates. It
was not independently authored and cannot support external-generalization or
clinical claims.

Run the complete scaffold with:

```bash
python scripts/run_finetune_scaffold.py
```

See `docs/finetuning_boundaries.md` for the hard safety and promotion contract.
