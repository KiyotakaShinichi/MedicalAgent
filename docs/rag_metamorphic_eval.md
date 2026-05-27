# RAG Metamorphic Eval

Status: internal derivative RAG robustness evidence.

This eval mutates the existing gold claim-grounding questions and checks that
the route/evidence policy remains stable under wording changes:

- plain wording
- polite framing
- Taglish prefix
- anxious framing
- hypothetical phrasing for refusal cases
- "I know you are not a doctor, but..." phrasing for refusal cases
- general education / care-team question phrasing for safe education cases

The eval checks:

- unsafe routes remain refusal/escalation routes
- safe education still routes through source-backed education or an allowed
  safe equivalent such as a care-team question summary
- education routes still require retrieval and claim validation
- no record-write tools execute during RAG/evidence-policy checks

Run:

```bash
python scripts/run_rag_metamorphic_eval.py
```

Artifact:

```text
Data/evals/rag/latest_rag_metamorphic_eval.json
```

Boundary: this is not an external-authored RAG benchmark and does not prove
clinical correctness or semantic entailment. It is meant to catch brittle
route/evidence-policy behavior before a reviewer sees the system.
