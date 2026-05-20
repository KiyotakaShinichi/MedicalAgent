# Safety Regression Response Runbook

1. Treat unsafe leakage as a hard blocker.
2. Locate the failing case and category.
3. Check whether the failure is pre-generation routing, retrieval, generation,
   claim validation, post-generation validation, or UI rendering.
4. Add a generalized rule or template change.
5. Add or update a focused unit test.
6. Rerun:

```bash
pytest tests/test_safety_invariants_property.py -q
python scripts/run_adversarial_safety_regression.py
python scripts/run_over_refusal_eval.py
python scripts/run_live_rag_eval.py
```

Do not tune directly to a hidden holdout set. Document contamination risk when a
case is used for tuning.
