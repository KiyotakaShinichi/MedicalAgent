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

The eval reports alias-normalized source matching because some older gold IDs
name a governed source concept while the current KB indexes generated chunk
IDs, parent IDs, and source titles. Treat raw-ID mismatch separately from true
retrieval failure.

Failure analysis:

```bash
python scripts/run_retrieval_failure_analysis.py
```

Artifact:

```text
Data/evals/rag/latest_retrieval_failure_analysis.json
```

The failure artifact classifies issues as retrieval, metadata/source-ID
normalization, source governance, scoring, or goldset design before tuning.
