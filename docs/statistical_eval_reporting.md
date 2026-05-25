# Statistical Eval Reporting

NLCare now reports uncertainty around internal engineering metrics instead of
only showing point estimates.

Run:

```bash
python scripts/run_statistical_eval_summary.py
```

Output:

```text
Data/evals/governance/latest_statistical_eval_summary.json
```

The report adds Wilson confidence intervals for pass rates and rough
fold-level intervals for model CV metrics.  It currently summarizes key
artifacts such as adversarial safety regression, held-out adversarial safety,
live RAG, retrieval goldset performance, and patient-level temporal CV.

This is still not clinical validation.  The intervals only describe internal
curated or synthetic engineering benchmarks.  Small sample sizes, internal case
authorship, benchmark contamination, and synthetic labels remain credibility
risks.

Reviewer rule:

- Show `n`, pass/fail/skipped counts, and CI beside every headline metric.
- Treat perfect scores as a reason to inspect coverage, not as proof of safety.
- Use external-author cases before making broader robustness claims.
