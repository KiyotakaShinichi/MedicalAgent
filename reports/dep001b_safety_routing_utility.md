# DEP-001B Safety Routing and Utility Calibration

**Status:** Internal blind targets passed, but formal completion is blocked by
a post-run freeze-integrity incident.  
**Clinical boundary:** Internal engineering evidence only. Not clinical
validation, not evidence of real-patient safety, and not production healthcare
readiness. DEP-001 remains blocked pending a new independently authored no-read
external holdout. The burned 400-case external bank was not rerun or used for
tuning.

## 1. External Failure Summary

The frozen official external evaluation contained 400 cases. It released no
unsafe outputs (`0/260`) and passed unsafe-recognition, language-parity,
multi-turn, and RAG-conditioned aggregate gates. It failed urgent routing
(`20/96`, `0.208333`) and safe educational utility (`86/140`, `0.614286`), with
over-refusal of `54/140` (`0.385714`). These aggregates motivated DEP-001B;
individual external prompts were not used as development examples.

## 2. Root Cause

The prior urgent probability was conditional on a separate risk-category
prediction. An urgent head prediction could therefore be zeroed or collapsed
into a generic unsafe refusal. Legacy adjudication could also override a valid
semantic urgent result. For safe education, medical vocabulary and policy
action were entangled: a semantic-safe result still passed through broad legacy
adjudication, allowing general dosing, interaction, or warning-sign education
to become a refusal.

The complete pre-change trace and failure taxonomy are documented in
`reports/dep001b_root_cause.md`.

## 3. Architecture Changes

DEP-001B separates five concepts:

1. calibrated unsafe-intent probability;
2. calibrated urgent-condition probability;
3. semantic intent family;
4. model uncertainty and disagreement;
5. deterministic policy action.

The action vocabulary is closed: `ALLOW_EDUCATIONAL`, `ALLOW_WITH_BOUNDARY`,
`SAFE_REDIRECT`, `REFUSE_ACTIONABLE`, `URGENT_ESCALATION`, and `FAIL_CLOSED`.
Safety-critical runtime, artifact, dependency, or component uncertainty fails
closed. RAG and generation cannot downgrade an earlier safety decision, and
the post-generation validator remains an independent output containment layer.

## 4. Urgent-Routing Design

Urgency is an independent decision dimension. Current, personal urgent
presentations receive `URGENT_ESCALATION` even when the unsafe head or family
head disagrees. A multilingual deterministic high-confidence lane covers
established present-tense danger descriptions; it does not diagnose. Generic
unsafe requests remain refusals or safe redirects when urgency is low.

## 5. Utility and Over-Refusal Design

The development corpus contains contrastive hard negatives that separate
general education from personalized action: dosing concepts versus personal
dose selection, treatment education versus treatment modification, interaction
education versus permission to start or stop a product, emergency-sign
education versus an active emergency, and tumor-marker education versus a
treatment decision. Bare symptom disclosure remains recordable, while requests
for self-management or treatment action remain bounded.

## 6. Intent-Family Architecture

The semantic family head distinguishes `EDUCATIONAL_GENERAL`,
`PERSONALIZED_INFORMATION`, `PERSONALIZED_ACTION_REQUEST`,
`TREATMENT_MODIFICATION`, `MEDICATION_DOSING`, `SYMPTOM_EDUCATION`,
`ACTIVE_SYMPTOM_MANAGEMENT`, `URGENT_PRESENTATION`,
`INTERACTION_EDUCATION`, `PERSONALIZED_INTERACTION_ACTION`,
`TUMOR_MARKER_EDUCATION`, `TUMOR_MARKER_TREATMENT_DECISION`, and
`UNKNOWN_HIGH_RISK`. The family is evidence for policy selection, not a medical
conclusion.

## 7. Policy Decision Matrix

| Condition | Policy action |
|---|---|
| Runtime/model/integrity failure | `FAIL_CLOSED` |
| High urgent probability or deterministic active danger | `URGENT_ESCALATION` |
| High-confidence actionable unsafe intent | `REFUSE_ACTIONABLE` or `SAFE_REDIRECT` |
| High-confidence general education | `ALLOW_EDUCATIONAL` or `ALLOW_WITH_BOUNDARY` |
| Unsafe/safe disagreement or ambiguous high risk | `SAFE_REDIRECT` |

Urgency has precedence over generic refusal. A high-confidence educational
family may permit bounded education, but cannot override deterministic
actionable or active-danger evidence.

## 8. Development Corpus

The new compositional corpus contains 8,280 cases: 5,600 train, 1,340
validation, and 1,340 internal test. A separate 750-case internal blind bank is
withheld from fitting, calibration, threshold selection, and runtime-assurance
development. It covers English, Taglish, and Filipino-heavy forms, hard safe
controls, urgent and nonurgent contrasts, indirect and fragmented messages,
multi-turn contexts, and RAG-conditioned requests.

The overlap audit reports zero exact overlap with prior DEP-001A development,
prior validation, and the burned external bank. Semantic overlap is reported
only as aggregate contamination diagnostics; the largest blind-to-burned-bank
similarity was `0.847702`, with no blind case at or above `0.90`.

## 9. Calibration

Separate logistic heads are calibrated over frozen multilingual MiniLM
embeddings. The validation unsafe head reports AUROC/AUPRC `1.0`, Brier
`0.000116`, and ECE `0.002009`; the urgent head reports AUROC/AUPRC `1.0`, Brier
`0.000232`, and ECE `0.001143`. These unusually strong results reflect a
synthetic compositional corpus and must not be read as external generalization.

## 10. Internal Validation

The 1,340-case internal test reports:

| Metric | Result | Target |
|---|---:|---:|
| Unsafe released outputs | 0 | 0 |
| Unsafe-intent recall | 1.000000 | >= 0.97 |
| Urgent-escalation recall | 1.000000 | >= 0.97 |
| Safe educational acceptance | 0.998936 | >= 0.95 |
| Over-refusal | 0.001064 | <= 0.05 |
| EN/Taglish unsafe-recall gap | 0.000000 | <= 0.03 |
| All-case pass rate | 0.999254 | informational |

One safe internal-test case was conservatively over-refused. Targets still
pass, but this error remains visible.

## 11. Language Slices

English, Taglish, and Filipino-heavy unsafe recall are each `1.000000` on the
internal test. This is internal generator-shaped evidence, not proof of broad
language generalization.

## 12. Multi-Turn Results

Internal-test multi-turn unsafe recall is `1.000000`. Conversation state is
bounded and earlier high-risk context cannot be silently discarded by a short
follow-up. A new external bank must test this independently.

## 13. RAG-Conditioned Results

Internal-test RAG-conditioned unsafe recall is `1.000000`. Safety action is
selected before retrieval, malicious or empty retrieval cannot relax it, and
downstream action merging is monotonic by safety rank.

## 14. Safe-Control Results

The internal safe-control acceptance rate is `0.998936`; over-refusal is
`0.001064`. Safe controls include medical vocabulary, symptom disclosures,
portal operations, emotional support, research education, and out-of-domain
requests. Allowing these controls does not bypass post-generation containment.

## 15. Fault Injection

All 13 injected failures passed containment: missing, corrupt, or stale safety
artifacts; disabled semantic safety; classifier failure; LLM timeout; empty or
malicious retrieval; validator exception; malformed patient context; invalid
cache payload; and dependency unavailability. No unsafe candidate or malicious
retrieval content was released.

## 16. Containment Regression

The focused DEP-001B and breast-monitoring regression suite passed 136 tests.
Unsafe released outputs remained zero. The existing post-generation validator,
evidence envelope, source policy, and refusal/escalation boundaries were not
weakened to improve educational acceptance.

## 17. Remaining Limitations

- The development and blind banks share an internal compositional authoring
  process; the blind result cannot be called independent external evidence.
- Near-perfect head metrics may reflect generator separability.
- One safe internal-test case remains over-refused.
- Deterministic multilingual urgency coverage is necessarily incomplete.
- No clinician reviewed the urgent triggers or response wording.
- No real patient data, clinical labels, IRB approval, or clinical validation
  exists.

## 18. New External-Holdout Readiness

The one-shot 750-case internal blind reported zero unsafe releases, unsafe and
urgent recall `1.0`, safe educational acceptance `1.0`, over-refusal `0.0`, all
three language unsafe recalls `1.0`, language gap `0.0`, and multi-turn and
RAG-conditioned unsafe recall `1.0`. Wilson 95% lower bounds were `0.984246`
for unsafe recall, `0.886487` for urgent recall, `0.992524` for safe acceptance,
`0.954182` for each language recall, and `0.862024` for multi-turn and
RAG-conditioned recall. The upper bounds were `0.015754` for unsafe-pass rate
and `0.007476` for over-refusal.

However, a timed-out overlap-test child process replaced one frozen evidence
artifact during the run. Start-time integrity passed and the changed file did
not affect scoring, but post-run verification is only 25/26. Per the declared
protocol, DEP-001B is therefore not formally complete and is not yet ready for
a new external holdout. The blind result is retained but must not be rerun or
used for tuning. See `reports/dep001b_freeze_integrity_incident.md`.
