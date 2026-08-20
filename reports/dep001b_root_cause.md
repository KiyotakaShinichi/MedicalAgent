# DEP-001B Root-Cause Analysis

## Scope and evaluation integrity

This analysis uses only the frozen official aggregate results, broad failure
classes declared in the DEP-001B brief, existing repository code, and newly
authored diagnostic probes. The burned 400-case external bank was not rerun,
opened for remediation, copied, relabeled, or used as development data.

DEP-001 remains blocked. This is engineering safety-routing analysis, not
clinical validation or evidence of real-patient safety.

## External failure summary

- Unsafe release containment held: 0/260 released unsafe outputs.
- Unsafe recognition passed its declared floor: 253/260 (0.973077).
- Urgent escalation failed: 20/96 (0.208333).
- Safe educational acceptance failed: 86/140 (0.614286).
- Over-refusal failed: 54/140 (0.385714).
- English/Taglish parity, multi-turn unsafe recognition, and RAG-conditioned
  unsafe recognition passed their declared aggregate gates.

The result therefore does not support broader unsafe blocking. It supports
separating risk recognition from policy-action selection.

## Decision path traced

The evaluated path is:

1. `multilingual_semantic_safety.classify_multilingual_safety`
2. calibrated unsafe and urgent binary heads plus a multiclass risk-category head
3. `unsafe_intent_semantic_classifier` and bounded conversation context
4. `safety_uncertainty_adjudicator.adjudicate_safety_uncertainty`
5. `agent_safety.safety_scope_check`
6. `agent_input_gate.input_guardrail_check`
7. `agent_intent_router.route_intent`
8. direct-support or RAG branch selection
9. post-generation validation and evidence-release enforcement

The external evaluator measures urgent recognition from the safety scope and
safe acceptance from the pre-generation safety and input-guardrail envelopes.
Consequently, RAG generation and post-generation validation are not the primary
causes of the two failed metrics. They remain essential containment layers.

## Root cause of urgent-routing failure

### 1. The urgent head is not independent

In `multilingual_semantic_safety._aggregate_turn_state`, an urgent probability
is zeroed unless the separate category head first predicts one of three urgent
categories. Even `URGENT_PROBABILITY_OVERRIDE` is nested inside that category
condition, so it cannot override a non-urgent category prediction.

This turns the intended two-dimensional unsafe/urgent surface into a serial
dependency:

`urgent head -> category head must agree -> urgent route`

An urgent presentation classified as `multi_turn_escalation`, medication,
toxicity, or another unsafe family can therefore be recognized as risky while
losing the urgent route.

### 2. Legacy unsafe adjudication can override semantic urgency

In `agent_safety.safety_scope_check`, when semantic safety requires a safe
route, a legacy adjudication scope is preferred whenever that adjudicator also
flags risk. This precedence applies even when semantic status is `urgent`.
The result can collapse `URGENT_ESCALATION` into a generic treatment or
diagnosis refusal.

### 3. The old corpus couples category and urgency

The DEP-001A corpus defines urgency through a small set of urgent risk
categories. Other categories, including multi-turn escalation, are normally
nonurgent. Threshold selection and validation reuse the same category gate,
so the reported internal urgent score tested the coupled rule rather than the
urgent head independently.

The configured urgent threshold is 0.021, but that low threshold does not help
when the category head vetoes the urgent probability. The external collapse
from strong internal reporting to 0.208333 is consistent with category-shape
dependence and policy override, not merely a slightly wrong threshold.

### 4. Response composition is downstream of the measured miss

The official evaluator marks urgency from `safety.scope` before generation.
The urgent failure therefore occurs before response composition, RAG, and the
post-generation validator. Those layers preserved zero unsafe release but
could not repair the route label.

## Root cause of over-refusal

### 1. Safe semantic evidence is not authoritative for action selection

Even after the multilingual semantic model returns `safe`,
`safety_scope_check` invokes the legacy uncertainty adjudicator. That
adjudicator can convert a safe general educational question into a high-risk
scope based on vocabulary/action-pattern composition.

Newly authored probes confirmed this path: all six new urgent probes routed
urgently, but two of eight general educational probes were converted to a
treatment-decision scope despite semantic status `safe`.

### 2. Medical vocabulary and policy action remain entangled

The legacy semantic classifier and adjudicator recognize protected topics and
action cues, but there is no independent semantic intent-family head for the
distinctions DEP-001B requires. A question about how dosing works generally can
share vocabulary with a request to choose a personal dose. Similar collisions
exist for interactions, warning signs, tumor markers, and treatment changes.

### 3. Uncertainty is treated as refusal pressure

`SemanticSafetyPrediction.requires_safe_route` includes every `uncertain`
result. The rescue path is narrow and depends on the legacy classifier returning
family `none` plus specific safe-boundary conditions. Natural educational
questions without explicit disclaimers can therefore fail closed into refusal
even when a bounded educational answer would be allowed.

### 4. Deterministic rules run before calibrated semantic intent

Deterministic treatment and diagnostic phrase checks execute before the
multilingual semantic model. They are useful high-confidence controls, but some
patterns are broader than the educational/actionable distinction. This allows
general medical wording to preempt an otherwise safe semantic result.

### 5. Internal validation was generator-shaped

DEP-001A reports high internal safe acceptance, while the independent external
result is 0.614286. Train and validation cases were produced by related template
families, often with explicit non-personal wording. The mismatch indicates that
the internal split measured template interpolation more strongly than natural
educational/actionable separation.

## Failure classification

Urgent cases are primarily:

- B: recognized as urgent by one signal but overridden by policy precedence;
- C: collapsed into generic unsafe refusal;
- F: affected by intent-family/category classification;
- G: affected by policy precedence;
- E: affected by coupled threshold calibration.

Safe educational failures are primarily:

- A: false unsafe/actionable classification;
- C: deterministic or compositional false positives;
- D: missing educational intent-family distinction;
- E: vocabulary correlation;
- F: possible history-risk carryover for short follow-ups;
- G: pre-RAG safety routing, not retrieval itself.

Post-generation validation is not the source of the measured over-refusal, but
it must remain independent to preserve zero unsafe release.

## Required remediation

DEP-001B must introduce an explicit deterministic policy layer over separate,
calibrated signals:

- unsafe-intent probability;
- urgent-condition probability;
- semantic intent family;
- uncertainty;
- policy action.

Urgency must have precedence over generic unsafe refusal. High-confidence
general education must be allowed with the existing medical boundary. Unknown
high-risk or runtime-failure states must remain fail closed. RAG, caching,
provider availability, and post-generation behavior must never weaken the
selected safety action.

The old external bank is permanently diagnostic-only. All training,
calibration, validation, and internal blind evidence for DEP-001B must be newly
constructed and hash-separated from that bank.
