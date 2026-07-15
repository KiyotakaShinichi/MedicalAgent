# Route-Aware RAG Policy Diagnostic

This diagnostic asks whether NLCare can preserve source governance on sensitive
routes while using a simpler retrieval path for ordinary education and portal
help. It composes the existing per-case frozen internal-goldset results:

- `education` and `portal_help`: BM25-only
- all safety-sensitive and medical-boundary routes: full source-governed stack

The policy is **post-hoc**. It was defined after the internal goldset and
baseline results were visible. The artifact therefore sets
`was_used_for_tuning: true`, does not modify live patient retrieval, and cannot
be used as held-out evidence. Promotion remains blocked until the no-read
external-author holdout is completed.

Run:

```powershell
python scripts/run_route_aware_rag_policy_eval.py
```

Artifact: `Data/evals/rag/latest_route_aware_rag_policy_eval.json`.

This is engineering evidence only. It is not clinical validation, a patient
benefit claim, or proof of production healthcare readiness.
