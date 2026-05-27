# Eval Credibility Audit

Status: benchmark transparency layer, not model validation.

The eval credibility audit scans release-gate artifacts and reports whether
they expose the metadata reviewers need to interpret internal scores:

- n-size or case-count metadata
- pass/fail/skipped counts
- authorship/provenance metadata
- tuning or contamination disclosure
- claim-boundary text
- explicit `clinical_validation: false`
- perfect-score caution flags for internal artifacts
- external-author detection

Run:

```bash
python scripts/run_eval_credibility_audit.py
```

Artifact:

```text
Data/evals/governance/latest_eval_credibility_audit.json
```

This audit intentionally does not turn internal metrics into external
evidence. A high number of perfect internal scores or missing contamination
disclosures should be treated as a credibility warning, not hidden.
