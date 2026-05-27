# Claim-Source Alignment Ledger

Status: internal goldset traceability artifact.

The claim-source alignment ledger makes the RAG goldset easier to inspect at
the claim level. For each gold case it emits rows for:

- supported claims that must keep an expected source ID and source-tier policy
- unsupported claims and contradiction traps that must be blocked or refused
- category-level pass/fail counts
- source ID traceability rate
- blocked-claim detection rate

Run:

```bash
python scripts/run_claim_source_alignment_eval.py
```

Artifact:

```text
Data/evals/rag/latest_claim_source_alignment_eval.json
```

Boundary: this is an offline engineering ledger over internal gold cases. It
does not prove clinical truth, real-world factuality, or clinician-approved
medical entailment.
