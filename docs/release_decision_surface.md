# Release decision surface

The v2 surface groups evidence into `aie`, `swe`, `mle`, `medical`,
`automation`, `fine_tuning`, and `deployment`. Each check is labelled as
verified internal evidence, needs attention, scaffolded, externally blocked,
missing, or invalid. Prepared packets and dry-run scaffolds therefore cannot
read like completed validation.

The full release registry remains the detailed evidence index. The compact
decision surface separates four hard engineering blockers from warnings and
informational review-readiness artifacts so a large optional artifact count
cannot visually dilute a critical failure.

Run:

```bash
python scripts/run_release_decision_surface.py
```

`PROCEED_WITH_WARNINGS` means only that the selected hard engineering checks
are present and acceptable. It does not mean the prototype is clinically
validated, clinician-approved, safe for real patient care, or production
healthcare ready.
