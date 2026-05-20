# Synthetic data quality proxy

A small set of internal-consistency checks on the synthetic patient
journeys. The output JSON labels itself
`synthetic_generator_quality_proxy` and includes an explicit
disclaimer in every report:

> These metrics measure internal consistency of the synthetic
> generator's output. They are NOT a measure of clinical realism and
> they do NOT establish that the data resembles any real patient
> population.

The test suite enforces that the disclaimer cannot be silently
weakened.

## What the report contains

- **Feature distributions**: per feature, the n_observed, n_missing,
  missing_rate, min, max, mean, std, and the count of values outside
  the hand-curated plausibility range.
- **Lab plausibility**: hand-curated min/max windows for each
  numeric feature in [`backend/services/synthetic_data_quality.py`](../backend/services/synthetic_data_quality.py).
  An out-of-range count > 0 means the generator emitted at least one
  value outside a physiologically reasonable window — not necessarily
  a bug, but worth a reviewer's attention.
- **Correlation preservation**: a few expected positive correlations
  (`pre_wbc` ↔ `pre_anc`, etc.) with an `expected_min_pearson`
  threshold; the report records observed Pearson and whether the
  threshold was met.

## Files

- Module: [`backend/services/synthetic_data_quality.py`](../backend/services/synthetic_data_quality.py)
- Script: [`scripts/run_synthetic_data_quality.py`](../scripts/run_synthetic_data_quality.py)
- Output: [`Data/evals/realism/latest_synthetic_data_quality.json`](../Data/evals/realism/latest_synthetic_data_quality.json)
- Compatibility alias: [`Data/evals/models/latest_synthetic_data_quality_report.json`](../Data/evals/models/latest_synthetic_data_quality_report.json)
- Tests: [`tests/test_synthetic_data_quality.py`](../tests/test_synthetic_data_quality.py)

## What this is NOT

- Not a measure of clinical realism. The ranges are hand-curated
  sanity windows.
- Not a measure of distribution similarity to any real patient
  cohort.
- Not a release gate signal on its own. A reviewer should read the
  report and decide whether the generator needs revising.
