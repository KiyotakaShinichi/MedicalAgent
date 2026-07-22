# TCIA I-SPY2 external engineering stress benchmark

## Scope

This benchmark uses the official TCIA I-SPY2 clinical spreadsheet and
multi-feature MRI spreadsheet as a separate public-data engineering bridge.
The source files are checksum-locked and the canonical export replaces the
trial subject identifier with a one-way local case key.

It does **not** train or validate NLCare's synthetic monitoring heads. The
external outcome is pathologic complete response (pCR), which is not the same
task as NLCare's longitudinal response-pattern, response-score, or toxicity
review signals.

Official source: <https://www.cancerimagingarchive.net/collection/ispy2/>

## Protocol

- Join 985 clinical rows to 384 multi-feature MRI rows.
- Exclude treatment arm from the canonical export and every feature set.
- Compare logistic regression with gradient boosting over five fixed,
  repeated stratified 75/25 splits.
- Compare clinical-only, baseline-MRI, and early-change feature sets.
- Report AUROC, average precision, Brier score, ranges across seeds, and paired
  feature-set deltas with bootstrap intervals.

## Boundaries

The artifact is public-data pipeline and model stress evidence only. It is not
clinical validation, does not authorize model promotion, is not patient-facing,
and does not establish safety, treatment utility, or patient benefit.
