# Uncertainty Dossier

Synthetic-only uncertainty documentation. No clinical uncertainty validation is claimed.

## response_classification
Method: calibrated probability plus evidence-aware abstention
Limitations: synthetic labels, rule-first sufficiency, external labels do not match

## response_score_regression
Method: quantile/conformal-style synthetic interval scaffold
Limitations: MRI percent-change target is simulator-defined, interval coverage is synthetic-only

## toxicity_review_signal
Method: review-priority signal with shortcut audits
Limitations: legacy target is nadir-CBC shortcut-prone, not CTCAE diagnosis

## abstention_sufficiency
Method: minimum evidence rules plus learned-abstention experiments
Limitations: engineering-authored thresholds, needs clinician review before relaxation
