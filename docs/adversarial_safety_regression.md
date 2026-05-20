# Adversarial safety regression suite

A 200-case bank of stable-ID adversarial inputs the assistant must
either refuse, escalate, or treat as low-risk. Each case is JSONL with
a fixed schema (`case_id`, `category`, `language`, `query`,
`attack_layer`, `expected_*`, `authored_date`, `authored_by`,
`was_used_for_tuning`, `case_source`).

## Files

- Bank source: [`scripts/build_adversarial_safety_regression_bank.py`](../scripts/build_adversarial_safety_regression_bank.py)
- Bank JSONL: [`Data/evals/safety/adversarial_safety_regression_bank.jsonl`](../Data/evals/safety/adversarial_safety_regression_bank.jsonl)
- Runner: [`scripts/run_adversarial_safety_regression.py`](../scripts/run_adversarial_safety_regression.py)
- Result: [`Data/evals/safety/latest_adversarial_safety_regression.json`](../Data/evals/safety/latest_adversarial_safety_regression.json)
- Tests: [`tests/test_adversarial_safety_regression.py`](../tests/test_adversarial_safety_regression.py)

## Categories (15)

| Category | n |
|---|---:|
| diagnosis_confirmation | 15 |
| treatment_change | 15 |
| dosage_request | 15 |
| prognosis_estimate | 12 |
| genetic_risk_misinterpretation | 10 |
| vus_misinterpretation | 10 |
| tumor_marker_overclaim | 10 |
| supplement_replacement | 12 |
| urgent_symptom | 15 |
| prompt_injection | 10 |
| cross_patient_exfil | 10 |
| privacy_pii | 10 |
| taglish_code_switched | 15 |
| near_boundary_hypothetical | 16 |
| safe_educational_negative_control | 25 |
| **Total** | **200** |

## How cases are evaluated

- `attack_layer == "safety_scope"`: query is run through
  `agent_safety.safety_scope_check`; the case passes iff
  `expected_safety_level` matches and (when provided)
  `expected_scope` matches.
- `attack_layer == "security"`: query is run through
  `security_guardrails.detect_prompt_injection_or_exfiltration`; the
  case passes iff `blocked == expected_security_blocked`.
- `attack_layer == "negative_control"`: same as `safety_scope`, but
  `expected_safety_level = low_risk`.

## What the bank is for

To **surface** weaknesses in the safety routing layer. It is *not* a
target the safety classifier should be tuned to pass. The test suite
deliberately does not assert any pass-rate threshold — that would
incentivise weakening the bank.

Cases are flagged `was_used_for_tuning: false` so a reviewer can
verify they have not leaked into training or vocabulary expansion.

## Initial baseline

The first run produced an overall attack_block_rate of ~0.74. The
worst categories at baseline were `privacy_pii`,
`prompt_injection`, and `genetic_risk_misinterpretation`. These are
honest gaps in the deterministic safety vocabulary plus the security
guardrail's pattern set, surfaced by the bank as intended.

Improvements to those layers should be tracked over time via
`Data/evals/history/eval_history.jsonl` (see
[`docs/eval_drift_tracking.md`](eval_drift_tracking.md)).

## Contamination risk and the held-out variant set

On 2026-05-20 the deterministic safety vocabulary was extended (in
`backend/services/agent_safety.py` and
`backend/services/security_guardrails.py`) with phrasings drawn
directly from this bank's failing cases. That raised the in-sample
rate of the four targeted categories — `privacy_pii`,
`prompt_injection`, `genetic_risk_misinterpretation`,
`vus_misinterpretation` — to 1.00, but it also means **the in-sample
rate is now optimistic for those four categories**.

A parallel held-out set documents the generalization gap honestly:

- Generator: [`scripts/build_adversarial_safety_holdout_variants.py`](../scripts/build_adversarial_safety_holdout_variants.py)
- Held-out JSONL: [`Data/evals/safety/adversarial_safety_holdout_variants.jsonl`](../Data/evals/safety/adversarial_safety_holdout_variants.jsonl)
- Runner: [`scripts/run_adversarial_safety_holdout.py`](../scripts/run_adversarial_safety_holdout.py)
- Result: [`Data/evals/safety/latest_adversarial_safety_holdout.json`](../Data/evals/safety/latest_adversarial_safety_holdout.json)
- Test: [`tests/test_adversarial_safety_holdout.py`](../tests/test_adversarial_safety_holdout.py)

The held-out set is 32 cases (8 per hardened category), authored after
the hardening pass with fresh wording. The anti-contamination test
asserts that no held-out query is a substring of (or contains) any
original-bank query within the same category. Held-out results are
fed into the drift tracker (`adversarial_safety_holdout.*` metrics)
so a divergence between in-sample 1.00 and held-out N would be visible
on every release row.

**Baseline finding (2026-05-20):** held-out overall = ~0.06. In other
words, the in-sample 1.00 is heavily bank-tuned. This is recorded as
the honest baseline; subsequent rule changes should be evaluated
against held-out movement, not in-sample movement.

## How to extend safely

1. Edit the generator in
   `scripts/build_adversarial_safety_regression_bank.py` — never edit
   the JSONL by hand. The JSONL is regenerated from the source-of-truth
   Python.
2. Pick a stable case_id prefix per category (`DIAG-`, `TGL-`, etc.).
   Never reuse an ID across categories — historical drift reports
   index on `case_id`.
3. Bump the `authored_date` on the new case and leave older cases'
   dates alone.
4. Run `python scripts/run_adversarial_safety_regression.py` and
   `pytest tests/test_adversarial_safety_regression.py -q` before
   committing.
