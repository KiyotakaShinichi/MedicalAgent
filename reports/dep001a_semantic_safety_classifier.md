# DEP-001A Multilingual Semantic Safety Classifier

**Status:** Internal runtime candidate ready for a new external-human no-read holdout.  
**DEP-001 status:** BLOCKED.  
**Clinical validation:** False.  
**Healthcare production readiness:** False.

## Root cause

The prior held-out failure was primarily an unsafe-intent recognition and
routing failure, not evidence that unsafe generated text reached the user. The
pre-generation path depended on deterministic phrases plus token/prototype
similarity. Its uncertainty adjudicator consumed the same weak semantic signal,
so paraphrases, Filipino-heavy Taglish, implied actions, and progressive turns
could remain low risk. Urgent recognition was also too vocabulary-sensitive.

Retrieval and generation were not the initiating cause: the previous final
output scorer released zero unsafe canaries, but the route-level recall and
urgent recall were unacceptable. That distinction matters. Downstream
containment reduced harm in those tested cases; it did not make recognition
generalize.

## Remediation

DEP-001A adds a frozen multilingual sentence encoder
(`paraphrase-multilingual-MiniLM-L12-v2`) with three linear heads:

- calibrated unsafe-intent probability;
- calibrated urgent-intent probability;
- semantic risk-family classification.

Platt scaling uses a disjoint calibration partition inside the development
bank. Thresholds are selected on the separate internal validation bank with
unsafe/urgent recall, language floors, multi-turn recall, and over-refusal
constraints. The word/character TF-IDF logistic model remains as a visible
lexical baseline; the semantic architecture is not presented as superior on
this synthetic grammar.

The live policy remains layered:

1. deterministic high-confidence urgent/action/diagnosis/security controls;
2. calibrated multilingual semantic classification;
3. legacy action-target semantic adjudication as an independent fallback;
4. bounded refusal, clarification, or escalation;
5. independent post-generation validation and final release authorization.

RAG cannot override the safety route. Missing, corrupt, stale, mismatched, or
unavailable classifier artifacts fail closed before retrieval or generation.
Urgent wording remains a separate calibrated head.
Urgent routing also requires agreement from the independently trained urgent
risk-family head. Unsafe routing uses the calibrated binary head or an unsafe
risk-family prediction at a fixed `0.60` confidence floor. This two-head
ensemble closed internal binary-head misses without allowing a safe adjudicator
to override a high-confidence unsafe result.

## Structured multi-turn state

Each user turn is embedded and classified independently. The runtime preserves
the maximum recent risk with bounded decay over the last four user turns. It
does not concatenate arbitrary conversation text into a hidden classifier
prompt. The trace records risk family, calibrated probabilities, uncertainty,
selected turn offset, model version, and context-turn count.

## Independent internal corpus

- Development: 4,600 cases.
- Validation: 1,150 cases.
- Languages: English, Taglish, Filipino-heavy Taglish.
- Families: safe education/information, ambiguous wording, personal medical
  action, treatment change, dosage, interactions, urgent deterioration,
  emergency symptoms, natural-language urgent descriptions, refusal bypass,
  role-play, hypothetical requests,
  authority impersonation, RAG-conditioned attacks, progressive multi-turn
  escalation, and indirect/euphemistic requests.
- Perturbations: punctuation removal, misspellings, fragments, benign
  preambles, slang, code-switching, vague wording, context poisoning, and
  conflicting intent.

The bank is internally and compositionally authored. It is useful for model
development and regression testing, not independent evidence and not clinical
validation. The old frozen final holdout was not opened for tuning or rerun.
The one-way integrity diagnostic emits only hashes and overlap counts.

## Results

Internal validation, before any new external holdout:

| Metric | Result |
|---|---:|
| Semantic ensemble unsafe recall | 0.9957 |
| Layered unsafe-intent recall | 1.0000 |
| Layered unsafe pass rate | 0.0000 |
| Urgent escalation recall | 0.9800 |
| Safe educational acceptance | 0.9600 |
| Over-refusal | 0.0400 |
| English unsafe recall | 0.9874 |
| Taglish unsafe recall | 1.0000 |
| EN/Taglish gap | 0.0126 |
| Adversarial paraphrase floor | 0.9600 |
| Multi-turn safety recall | 1.0000 |
| RAG-conditioned safety recall | 1.0000 |
| Runtime canary pass rate | 1.0000 |
| Fault-injection pass rate | 1.0000 (12/12) |
| Unsafe output releases | 0 |

Wilson 95% confidence intervals are machine-readable in
`Data/evals/safety/dep001a/latest_runtime_assurance.json`. A point estimate of
1.0 does not mean the true rate is certainly 1.0, especially for the smaller
urgent and safe-control slices.

## Fault injection

The tested paths include classifier failure, provider timeout, empty retrieval,
malicious retrieved context, validator exception, malformed patient context,
invalid cache payload, dependency failure, missing semantic artifacts, corrupt
semantic artifacts, stale semantic artifacts, and malformed structured context.
All tested paths returned a safe route or blocked unsafe output.

## Remaining limitations

- DEP-001 remains blocked until a fresh eligible external human authors a
  no-read final bank after this implementation freeze.
- The 4,600/1,150 corpus is generated from an internal threat grammar and may be
  easier than naturally occurring language.
- The lexical baseline is perfect on the current synthetic validation grammar;
  this demonstrates that the bank still contains learnable surface regularity.
- The semantic head has non-zero model-only misses; layered defenses catch the
  current internal misses, but that may not transfer to external language.
- No clinician, nurse, pharmacist, or genetic counselor reviewed the labels or
  response wording.
- No real patient data, patient benefit, clinical safety, or deployment claim is
  supported.

## Freeze and next evaluation

Freeze the current code, model/calibration/threshold hashes, corpus manifests,
and this report. Then commission a **new external-human no-read holdout**. The
author must not inspect production prompts, classifiers, threat grammar,
validation cases, failures, or the old final holdout. Any unsafe final output on
that new bank keeps DEP-001 blocked; route, urgency, parity, multi-turn,
RAG-conditioned safety, and over-refusal must also be reported separately.
The machine-readable freeze is
`Data/evals/safety/dep001a/implementation_freeze_manifest.json`; external-author
instructions are in `reports/dep001a_external_no_read_protocol.md`.
