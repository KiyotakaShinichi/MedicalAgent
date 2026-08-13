# DEP-001 Root-Cause Analysis

Status: architectural remediation implemented; independent assurance gate blocked  
Scope: engineering safety assurance for a synthetic-only, non-diagnostic prototype  
Clinical validation: false

## Observed failure

`Data/evals/safety/latest_adversarial_generalization_v2_eval.json` reports a
held-out pass rate of `0.7818` and an `unsafe_leakage_rate` of `0.2182`.
Inspection of the evaluator shows that this number is produced by checking the
intent/safety classifier layer only. A route or safety-level mismatch is counted
as leakage even though the evaluator does not execute retrieval, generation,
post-generation validation, or the final evidence-release boundary.

This means the historical number is an important unsafe-intent routing signal,
but it is not a direct measurement of unsafe text released to a user. DEP-001
must report those two failure modes separately.

## Exact execution path

The patient support path is:

1. `support_chat_agent.handle_support_chat` normalizes the turn and loads prior
   user turns.
2. `agent_safety.safety_scope_check` applies deterministic urgent/medical
   patterns and the semantic unsafe-intent classifier.
3. `agent_input_gate.input_guardrail_check` independently blocks prompt
   injection, exfiltration, and security-boundary violations.
4. `agent_rag.run_patient_agent_pipeline` selects a direct-support or RAG route.
5. The RAG branch performs retrieval, source governance, reranking/context
   shaping, answerability assessment, and generation.
6. `_finalize_result` runs output guardrails, the post-generation validator,
   RAG governance, and the evidence-release boundary before returning a reply.
7. The outer pipeline exception boundary returns a deterministic fail-closed
   result when an unhandled dependency or component failure occurs.

The legacy v2 evaluator stops after step 2 (or the security helper used by step
3), so it cannot prove final-output leakage or final-output containment.

## Root causes

### 1. Uncertain unsafe classifications can fail open

`unsafe_intent_semantic_classifier` is a lexical/compositional prototype
classifier, not a general semantic encoder. It is sensitive to paraphrase,
obfuscation, indirect wording, and code switching. More importantly,
`safety_scope_check` promotes only non-borderline unsafe results above its high
confidence threshold. Borderline or lower-confidence risk can fall through to
the default low-risk, cache-eligible route.

### 2. Multi-turn uncertainty is discarded

The context combiner preserves confident unsafe intent but can drop borderline
signals from prior turns. A harmless-looking follow-up can therefore be judged
without enough weight from the unsafe setup turn.

### 3. Layer metrics are conflated

The existing evaluator labels classifier mismatch as `unsafe_leakage_rate`.
That obscures whether the system misrouted a request, released unsafe text, or
misclassified a safe educational request. Severe final-output failures must be
reported independently and must never be averaged away.

### 4. One validator primitive accepts malformed output

`post_generation_validator.validate_reply` currently returns `allowed` for an
empty or non-string reply. Other pipeline checks normally contain this, but the
primitive itself is fail-open and should be independently safe.

### 5. No frozen final-output fault gate covers the whole boundary

Existing banks exercise valuable classifier and policy behavior, but DEP-001
does not yet have a sealed split bank that tests routing and final released
output under classifier, provider, retrieval, validator, context, cache, and
dependency failures.

## Architectural remediation

The smallest robust change is a layered uncertainty adjudicator between the
prototype classifier and route selection. It uses a compact risk ontology
(requested action, protected medical/security object, personal/authority
context, and explicit educational boundary) rather than memorizing holdout
strings. Ambiguous safety-critical intent is sent to safe clarification or a
high-risk boundary route and is never cache eligible.

The existing deterministic security gate remains authoritative. RAG cannot
override it. Post-generation validation and the evidence-release boundary
remain independent final layers. Malformed validator input is changed to block.
Multi-turn context carries unresolved unsafe uncertainty forward until an
explicit safe reset is present.

## Evaluation correction

DEP-001 will publish separate metrics for:

- unsafe-intent routing recall;
- final-output unsafe pass rate;
- urgent escalation recall;
- safe educational acceptance and over-refusal;
- English/Taglish parity;
- paraphrase, multi-turn, RAG-conditioned, and injected-failure safety.

The deployment blocker is final-output `unsafe_pass_rate == 0`. The artifact
will also retain contamination/authorship metadata. Passing an internally
isolated bank remains engineering evidence only and does not substitute for an
eligible external no-read author or clinical review.

The corrected evaluator now invokes the real top-level
`run_patient_agent_pipeline` for every case with persistence and provider calls
disabled. It separately records the pre-generation route and the reply that
survives post-generation validation, evidence authorization, and transport
release. RAG-conditioned fixtures enter only through the evaluator's retrieval
boundary; production modules do not import the bank or evaluator.

## Reassessment

The frozen 180-case run found zero released unsafe canaries, but it did not
clear DEP-001. Unsafe-intent recall was `0.3727`, urgent-escalation recall was
`0.2000`, safe educational acceptance was `0.7800`, and EN/Taglish parity was
`0.8063`. This shows that downstream containment is stronger than upstream
recognition, while usefulness and multilingual consistency are below the
declared deployment thresholds. The frozen bank must not be used for tuning.
Any further classifier work must use development data, followed by a fresh
eligible no-read holdout.

## Claim boundary

This remediation can demonstrate stronger fail-closed engineering behavior. It
does not establish clinical validation, real-world safety, patient benefit,
clinician approval, or production healthcare readiness.
