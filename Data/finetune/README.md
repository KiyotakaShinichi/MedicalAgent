# NLCare fine-tuning scaffold

This directory is the **behavior / style** fine-tuning scaffold for
NLCare.  It does NOT contain:

- real patient data
- clinical knowledge tuning
- diagnostic / prognostic / dosage / treatment / genetic-risk /
  tumor-marker training examples

What it IS for:

- consistent patient-safe tone
- structured clinician-summary formatting
- non-diagnostic refusal language
- Taglish patient-safe explanations
- missing-data disclosure language
- "questions to ask your care team" generation
- safe supplement / genetics / tumor-marker boundary phrasing
- portal-help style responses

## File layout

    templates/                      ← five JSONL template files, each scoped to one behavior
        clinician_summary_examples.jsonl
        taglish_safety_examples.jsonl
        missing_data_disclosure_examples.jsonl
        questions_to_ask_care_team_examples.jsonl
        supplement_boundary_examples.jsonl

## Dataset card fields

Every prepared dataset emits these fields alongside the JSONL output:

- ``dataset_purpose``               — what behavior the dataset trains
- ``allowed_behavior_targets``      — the eight allowed targets above
- ``blocked_claims``                — diagnosis, treatment, dosage, etc.
- ``synthetic_or_source``           — origin of every example (always synthetic for now)
- ``safety_filters_applied``        — refusal-pattern + medical-claim-boundary scans
- ``known_risks``                   — over-fitting on synthetic style, etc.

## Model card stub fields

A future fine-tuned adapter must publish a model card with at least:

- base model
- adapter name / training run id
- intended behavior (one of the allowed targets)
- not intended for
- evaluation results
- safety validator compatibility

## Hard rule: post-tune deployment still goes through every safety layer

Any fine-tuned model that ships must still pass:

- deterministic safety gates (``agent_safety``, ``agent_input_gate``)
- source-governed RAG (tier filter + claim validator + evidence grade)
- the post-generation validator
- the medical claim boundary checker
- clinician review for new behavior

If the fine-tuned adapter weakens any of those layers, the candidate
must be rejected by the offline A/B framework (see ``config/ab_tests.yaml``
and ``backend/services/ab_testing.py``).

## Running the scaffold

Dry-run dataset preparation (no GPU required):

    python scripts/prepare_finetune_dataset.py --output-dir data/finetune/prepared

LoRA training dry-run (mocked — no real training):

    python scripts/run_lora_finetune_dryrun.py --dataset data/finetune/prepared/dataset.jsonl

Behavior evaluation against the prepared dataset:

    python scripts/evaluate_finetuned_behavior.py --dataset data/finetune/prepared/dataset.jsonl

All three scripts are deterministic and work offline.

## Boundary statement (repeat to be safe)

This scaffold does not train medical knowledge.  It trains tone, format,
and refusal style on synthetic examples whose claims are bounded by the
medical claim boundary checker.  Any deployment of a tuned adapter
remains a non-diagnostic clinician-in-the-loop engineering prototype.
