# QLoRA / Ollama Behavior Experiment

This is a learning scaffold only. It is not a medical-knowledge fine tune.

## Purpose

Tune behavior around:

- structured tool-intent JSON
- non-diagnostic patient support wording
- refusal and escalation
- insufficient-evidence responses
- citation-aware formatting

## Not Intended For

- memorizing oncology facts
- diagnosis
- treatment recommendation
- replacing RAG
- clinical validation

## Dataset

- `experiments\qlora_behavior\behavior_tuning_examples.jsonl`

## Evaluation Before Any Claim

- JSON validity rate
- refusal accuracy
- unsafe advice rate
- escalation accuracy
- summary completeness
- citation-format adherence

RAG remains the factual grounding layer. This experiment is only for behavior and format.
