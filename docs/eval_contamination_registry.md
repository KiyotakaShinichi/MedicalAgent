# Eval Contamination Registry

Status: benchmark provenance and contamination tracking.

The eval contamination registry scans RAG, safety, and bounded-agentic case
banks/artifacts and records:

- artifact path
- case count
- authored_by / authored_date values
- whether any case was used for tuning
- internal vs external-authored signals
- holdout/frozen/template classification
- contamination disclosure coverage
- recommended use: internal regression, holdout warning, or supporting artifact

Run:

```bash
python scripts/run_eval_contamination_registry.py
```

Artifact:

```text
Data/evals/governance/latest_eval_contamination_registry.json
```

Boundary: this makes internal eval limits louder. It does not turn internal
cases into external evidence. If a case bank was used for tuning, it should be
reported as regression evidence, not independent generalization evidence.
