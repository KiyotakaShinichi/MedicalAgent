# Portfolio / CV safe-wording template

> If a sentence about this project would be false to say in a
> courtroom or in front of a regulator, it must not be said in a CV,
> README, LinkedIn, or recruiter blurb either. When in doubt, say
> "engineering prototype" and "synthetic-only".

The machine-readable artifact is at
[`Data/evals/governance/latest_portfolio_claim_safety_check.json`](../Data/evals/governance/latest_portfolio_claim_safety_check.json).
This doc is the human reading version. The test suite enforces
that:

- this doc does NOT contain bare affirmative banned phrases;
- the artifact's `claim_boundary` says `not clinical validation`;
- the artifact lists the same banned + allowed phrases as this doc.

## Banned affirmative phrases

These must NEVER appear as bare claims (no "not", no "is not", no
"does not"). If they appear with a negation, they're fine as
disclaimers.

- clinically validated
- production healthcare ready
- patient benefit
- diagnostic system
- treatment recommender
- proven safe
- clinician-approved
- fhir compliant
- hospital interoperable
- fda approved / fda cleared / ce marked
- hipaa compliant
- real-world evidence

## Allowed phrases (safe under current constraints)

- engineering prototype
- synthetic-only ML signals
- not clinically validated
- non-diagnostic
- monitor-only
- intended for clinician review
- source-governed retrieval
- claim-level citation validation
- release-gate-enforced
- in-sample only
- improvement not proven
- informational artifact only

## Sample wordings by audience

### LinkedIn one-liner

✅ Safe:
> Built an engineering prototype of a safety-first breast cancer
> monitoring agent with source-governed RAG, claim-level citation
> validation, and release-gate-enforced negative-result reporting;
> synthetic-only data, not clinically validated.

❌ Unsafe:
> Built a clinically validated, production-ready AI doctor that
> diagnoses breast cancer using FHIR-compliant patient data.

Why unsafe: claims clinical validation, diagnosis authority, FHIR
compliance, and production readiness — none are true.

### Recruiter short

✅ Safe:
> Designed and shipped a synthetic-data oncology monitoring agent:
> hybrid RAG, source-tier governance, adversarial safety bank with
> held-out generalisation reported honestly, and a 120-artifact
> release gate with explicit anti-overclaim tests.

❌ Unsafe:
> Built an AI cancer agent that improves patient outcomes and
> supports clinical decision-making in hospitals.

### Senior engineer / technical

✅ Safe:
> RAG architecture with 5 intent-aware source-governed modes, hybrid
> dense+sparse RRF, query rewriting, claim-level citation validation
> (heuristic by default, NLI opt-in), uncertainty-aware
> answerability routing, per-turn trace with chain-of-thought
> deny-list, stage-wise oracle diagnostic. Source-governed stack
> does not exceed BM25 on raw recall on the in-sample goldset;
> negative results documented; held-out v2 prepared but not
> completed.

❌ Unsafe:
> RAG architecture that outperforms baselines on retrieval and is
> clinically validated for oncology decision support.

### README summary paragraph

✅ Safe:
> MedicalAgent is a safety-first, non-diagnostic breast cancer
> monitoring engineering prototype. It combines source-governed
> dense/sparse RAG, claim-level citation validation, deterministic
> pre-generation safety gates, adversarial safety regression with
> held-out generalisation reporting, and release-gate-enforced
> negative-result publication. All ML signals are synthetic and
> monitor-only. No clinician sign-off, no IRB, no real patient
> data.

❌ Unsafe:
> MedicalAgent is a clinically validated breast cancer monitoring
> system used in hospitals to improve patient outcomes.

### CV bullet

✅ Safe:
> Engineering prototype of a non-diagnostic oncology monitoring
> agent on synthetic data; documented negative results (pruner
> regression, held-out adversarial gap, full-stack not exceeding
> BM25); test-locked anti-overclaim invariants.

❌ Unsafe:
> Shipped clinically validated AI for breast cancer diagnosis used
> by oncologists.

## How to use this template

1. Start from the safe samples above.
2. Add specifics from the
   [10/10-under-constraints roadmap](ten_out_of_ten_under_constraints.md)
   if you need more depth (e.g., specific artifact names, ADR
   numbers).
3. Cross-check the wording against this doc's banned list; the
   test suite will catch most accidental claims if you keep
   editing the doc itself.
4. When in doubt, write less.

## Related

- [10/10-under-constraints roadmap](ten_out_of_ten_under_constraints.md)
- [Negative results gallery](negative_results_gallery.md)
- [README](../README.md) — current README already follows these
  constraints; treat it as a reference.
