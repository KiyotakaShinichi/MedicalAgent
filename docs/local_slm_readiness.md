# Local SLM Readiness

Local SLM support is optional engineering scaffolding for low-risk helper tasks.
It is not a clinical answering system and must not become the final authority for
medical decisions.

## Allowed Helper Tasks

- intent classification
- query rewriting
- claim extraction
- summary formatting
- portal help
- refusal-style drafting after the refusal decision has already been made

## Blocked Solo Tasks

Local SLMs must not independently answer or decide:

- diagnosis
- treatment advice
- prognosis
- dosage guidance
- genetic-risk interpretation
- tumor-marker interpretation
- medication safety
- supplement safety

## Required Gates

Any local SLM output must remain behind:

- deterministic pre-generation safety gate
- source-governed RAG filtering
- medical claim boundary checker
- claim-level citation validation
- post-generation safety validator
- release gate checks

Run:

```bash
python scripts/run_local_slm_readiness.py
```

Output:

```text
Data/evals/ops/latest_local_slm_readiness.json
```

This artifact documents which tasks are allowed, which are blocked, and which
guards must still run.
