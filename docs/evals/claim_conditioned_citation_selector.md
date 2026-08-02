# Claim-Conditioned Citation Selector

This evaluation-only component assigns already-governed retrieved chunks to
individual generated claims. It separates retrieval recall from citation
selection and leaves unsupported claims visible instead of attaching a merely
related source.

It is disabled on the live patient route. The current comparison is internally
authored and was used while developing the selector. A positive result can
therefore justify only an offline shadow candidate, not promotion.

The selector:

- excludes stale and clinician-only chunks;
- selects at most two sources per claim;
- emits no citations on refusal routes;
- reports unsupported claims explicitly; and
- labels lexical support as a proxy, not semantic or medical entailment.

Before any live A/B test, it must be evaluated on frozen generated answers with
claim-support review, contradiction cases, safe refusals, and citation assembly
latency. This is engineering evidence only, not clinical validation or
production healthcare readiness.
