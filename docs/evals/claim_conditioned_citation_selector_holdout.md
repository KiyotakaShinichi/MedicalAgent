# Claim-conditioned citation selector frozen holdout

This evaluation compares the existing top-three citation window with the
claim-conditioned selector on 45 pre-existing answer outputs and a frozen
governed retrieval snapshot. The fixture is marked
`was_used_for_selector_tuning: false` and cannot be overwritten without an
explicit `--force` freeze operation.

## Result

- Baseline citation precision: `0.0909`
- Selector citation precision: `0.0429`
- Delta: `-0.0480`
- Bootstrap 95% interval: `[-0.1061, -0.0051]`
- Baseline support proxy: `0.1818`
- Selector support proxy: `0.1212`
- Baseline unsupported-context proxy: `0.8182`
- Selector unsupported-context proxy: `0.8788`
- Promotion: `offline_only_not_promoted`

The selector preserved source governance and refusal citation stripping, but
that does not compensate for lower grounding metrics. It is not connected to
the live patient route.

## Measurement limitation

Many legacy answer rows do not contain an expected source ID. Those rows are
correctly scored as lacking gold support, which makes the absolute support and
precision values weak. The paired negative direction is still visible, but
this artifact is an internal answer-level diagnostic, not external evidence or
semantic entailment.

Run without changing the fixture:

```bash
python scripts/run_claim_conditioned_citation_selector_holdout.py
```

This result is not clinical validation and does not establish real-world
grounding, safety, or healthcare-production readiness.
