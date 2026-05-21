# RAG Retrieval Goldset

The retrieval goldset checks whether retrieval finds governed source evidence
before generation.

Fields include:

- query
- gold source IDs
- acceptable source tiers
- relevant chunk IDs when available
- contradiction traps
- near-duplicate distractors
- stale-source distractors
- clinician-only distractors
- expected allowed use
- expected refusal or insufficient-evidence behavior

Run:

```bash
python scripts/run_retrieval_goldset_eval.py
```

Artifact:

```text
Data/evals/rag/latest_retrieval_goldset_eval.json
```

Do not claim cross-encoder improvement unless `improvement_proven` is true.
Current metrics are retrieval engineering evidence only, not medical correctness.
