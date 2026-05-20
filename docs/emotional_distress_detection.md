# Emotional distress detection

Classifies the affective signal of a patient query into one of five
categories (`crisis`, `despair`, `fear`, `anxiety`, `denial`) plus
`none`. Selects a `response_mode` for the answer composer:

| Mode | When |
|---|---|
| `crisis_support` | crisis wording detected (self-harm, suicidal ideation) |
| `urgent_clinician_review` | despair AND safety scope already high-risk |
| `clinician_review_with_warm_handoff` | despair (no other high-risk signal) |
| `empathetic_support_plus_education` | fear, anxiety, or denial wording |
| `normal_education` | no affective signal above threshold |

The module is layered orthogonally to `agent_safety.safety_scope_check`:
a query can be safety low-risk and still need an empathetic mode
(e.g., "I'm so scared about my next scan"), and a query can be safety
high-risk and emotionally neutral (e.g., "Should I stop chemo?").

## Taglish parity

Every category has explicit Taglish vocabulary. The test suite
enforces this — a future contributor cannot remove all Taglish from
a category without failing
[`tests/test_emotional_distress_detection.py`](../tests/test_emotional_distress_detection.py).

## Files

- Module: [`backend/services/emotional_distress_detection.py`](../backend/services/emotional_distress_detection.py)
- Probe script: [`scripts/run_emotional_distress_eval.py`](../scripts/run_emotional_distress_eval.py)
- Eval JSON: [`Data/evals/safety/latest_emotional_distress_eval.json`](../Data/evals/safety/latest_emotional_distress_eval.json)
- Tests: [`tests/test_emotional_distress_detection.py`](../tests/test_emotional_distress_detection.py)

## What this layer is NOT

- Not a substitute for a clinical mental-health screener (PHQ-2/9 or
  similar). It is a wording-level detector that triggers a different
  response mode in chat; it does not produce a clinical risk score.
- Not exhaustive. Vocabulary is heuristic and English/Taglish only.
- Not validated against patient-reported outcomes.
