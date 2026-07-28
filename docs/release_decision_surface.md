# Release decision surface

The v3 surface groups evidence into `aie`, `mle`, `swe`,
`data_engineering`, `infrastructure`, `medical`, `automation`,
`deployment`, and `fine_tuning`. Each check is labelled as verified internal
evidence, needs attention, scaffolded, externally blocked, stale, missing, or
invalid. Prepared packets, old artifacts, and dry-run scaffolds therefore
cannot read like current completed validation.

The full release registry remains the detailed evidence index. The compact
decision surface separates four hard engineering blockers from warnings and
informational review-readiness artifacts so a large optional artifact count
cannot visually dilute a critical failure.

The primary surface is capped at 20 canonical checks. Superseded holdouts,
component diagnostics, and duplicate readiness artifacts remain discoverable
in the detailed registry, but they do not compete with the primary decision.

The ship workflow regenerates source artifacts before this surface, then builds
the cross-domain improvement program, benchmark registry, service-health
snapshot, focused summary, and final release gate in dependency order.

Run:

```bash
python scripts/run_release_decision_surface.py
```

`PROCEED_WITH_WARNINGS` means only that the selected hard engineering checks
are present and acceptable. It does not mean the prototype is clinically
validated, clinician-approved, safe for real patient care, or production
healthcare ready.
