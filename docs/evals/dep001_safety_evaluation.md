# DEP-001 Safety Evaluation

## Scope

DEP-001 measures engineering safety behavior for the synthetic NLCare
prototype. It is not clinical validation, proof of patient safety, clinician
approval, or production-healthcare readiness.

The evaluation distinguishes two failures that older reports conflated:

- **Route miss:** the unsafe-intent route was not selected.
- **Unsafe pass:** disallowed candidate output reached the release boundary.

An unsafe pass is always a deployment blocker. It is never averaged away by
safe cases or reported only through an aggregate pass rate. Each case executes
the real top-level patient-agent pipeline; persistence and external provider
calls are disabled so evaluation prompts cannot become application data or
reach a generation provider.

## Root Cause And Remediation

The detailed trace is in `docs/evals/dep001_root_cause.md`. The main issues
were confidence-threshold fall-through for uncertain semantic classifications,
loss of borderline unsafe state across turns, trust in caller/provider fallback
text on high-risk routes, and incomplete post-generation coverage.

The remediation adds an independent action/target uncertainty adjudicator,
contextual unsafe-state carryover, deterministic high-risk response composition,
independent output validation, malformed-context rejection, and fail-closed
top-level exception behavior. RAG content is never allowed to supersede the
safety envelope.

## Bank Separation

- `development_safety_bank.jsonl`: tuning-used engineering cases.
- `validation_safety_bank.jsonl`: initially separate, then explicitly marked
  tuning-used after its failures were inspected and remediated.
- `final_holdout_safety_bank.jsonl`: frozen after runtime remediation, not used
  for tuning, protected by SHA-256, and scored once.

The final bank was authored by configured LLMs under a no-read category-only
contract. Exact normalized overlap with existing evaluation material is zero.
It is still **not** an eligible independent human-authored evaluation, and its
labels have not been adjudicated by clinicians.

## Frozen Final Result

The final bank contains 180 cases: 10 in each of 11 unsafe families, 50 safe
educational controls, and 20 urgent-symptom cases.

| Metric | Result |
|---|---:|
| Unsafe final-output pass | 0 / 110 (0.0000; Wilson 95% CI 0.0000-0.0337) |
| Unsafe-intent route recall | 0.3727 (95% CI 0.2881-0.4659) |
| Urgent escalation recall | 0.2000 (95% CI 0.0807-0.4160) |
| Safe educational acceptance | 0.7800 |
| Over-refusal | 0.2200 (95% CI 0.1275-0.3524) |
| EN/Taglish parity | 0.8063 |
| Adversarial paraphrase robustness | 0.3086 |
| Multi-turn safety | 0.2340 |
| RAG-conditioned safety | 0.4545 |
| Fault-injection safety | 8 / 8 |

**Verdict:** failed. The deterministic release boundary prevented all tested
unsafe candidate outputs from escaping, but upstream intent and escalation
generalization is not deployable. Production logic must not be tuned on this
bank. A newly authored holdout is required after any future remediation.

The release artifact includes grouped failures by reason, family, language,
and style. It records route misses independently from unsafe final-output
passes, so a strong final validator cannot conceal weak first-pass routing.

## Fault Injection

The real top-level agent boundary was tested for classifier exception, LLM
timeout, empty retrieval, malicious retrieval, validator exception, malformed
patient context, invalid cache payload, and unavailable dependency. All eight
returned a refusal, abstention, boundary response, or internal-error envelope
without releasing the injected unsafe candidate or malicious context.

## Remaining Limitations

- No eligible external human author or reviewer has completed the protocol.
- No clinician, nurse, genetic counselor, or pharmacist has adjudicated labels.
- The bank is model-authored and may contain ambiguous or unnatural wording.
- Zero observed unsafe passes does not prove zero real-world risk; the upper
  confidence bound remains nonzero.
- This evaluation uses synthetic/test context only and says nothing about
  clinical outcomes or real-patient safety.
