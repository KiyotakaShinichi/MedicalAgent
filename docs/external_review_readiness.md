# External Review Readiness

NLCare now has packet and template checks for future external-author and expert review.

This is **not** completed external review. It only verifies that reviewers can be given clean instructions, blank templates, and log fields before they author cases or comments.

Current artifact:

```bash
python scripts/run_external_review_readiness.py
```

Output:

```text
Data/evals/governance/latest_external_review_readiness.json
```

The artifact tracks required packet files, template fields, reviewer roles, and whether any completed external review logs exist. Current expected status is readiness only: no clinician approval, no genetic-counselor review, no senior MLE sign-off, and no clinical validation.
