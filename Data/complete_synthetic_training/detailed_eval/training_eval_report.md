# Detailed Synthetic Training Evaluation Report

Generated from the current local synthetic training artifacts.

## Claim Boundary

Synthetic-data engineering evaluation only. This report helps visualize model behavior, rule routing, and error modes. It is not clinical validation.

## Headline Results

- Test patients: 105
- Best classifier: `gradient_boosting`
- Best regressor: `random_forest_regressor`
- Classifier patient-level AUROC: `0.995`
- Classifier patient-level AUPRC: `0.996`
- Classifier patient-level Brier score: `0.05`
- Calibrated champion status: `trained`
- Calibrated validation ECE: `0.0298`
- Regressor patient-level MAE: `2.921`
- Regressor patient-level RMSE: `11.673`
- Regressor patient-level R2: `0.915`

## Hybrid Ruleset

| rule_order | category | condition | reason |
| --- | --- | --- | --- |
| 1 | toxicity_review | (max_symptom_severity >= 8 AND (nadir_wbc < 1.0 OR nadir_anc < 0.5)) OR nadir_wbc < 0.5 OR nadir_anc < 0.2 OR nadir_plat | Only severe CBC/symptom patterns override favorable ML signal; routine chemo nadirs should not dominate every route. |
| 2 | discordant_signal_review | classification band conflicts with regression band | Classifier and continuous response estimate disagree. |
| 3 | response_trend_review | hybrid_score < 55 OR latest_mri_percent_change > -20 | Lower or weak response signal needs review. |
| 4 | watch_closely | 55 <= hybrid_score < 75 | Mixed signal; monitor closely. |
| 5 | routine_monitoring | hybrid_score >= 75 and no override rules | Favorable synthetic monitoring signal with no major rule trigger. |

## Hybrid Routing Summary

| hybrid_review_category | patients | mean_hybrid_score | mean_probability | mean_response_score | toxicity_rule_rate | response_review_rule_rate | patient_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| toxicity_review | 76 | 63.441 | 0.538 | 42.325 | 1.0 | 0.487 | 0.724 |
| routine_monitoring | 14 | 99.627 | 1.0 | 63.686 | 0.0 | 0.0 | 0.133 |
| discordant_signal_review | 9 | 37.71 | 0.102 | 41.065 | 0.0 | 0.889 | 0.086 |
| response_trend_review | 6 | 10.279 | 0.0 | -20.632 | 0.0 | 1.0 | 0.057 |

## Example Test-Set Predictions

| patient_id | actual_label | champion_probability | actual_response_score_percent | champion_response_score_percent | hybrid_score | model_agreement | hybrid_review_category | rule_explanation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COMPV4-BRCA-0004 | 0 | 0.0 | -10.72 | -10.744 | 13.74 | aligned | toxicity_review | CBC/symptom toxicity rule; low hybrid or weak MRI improvement |
| COMPV4-BRCA-0006 | 1 | 1.0 | 91.18 | 91.535 | 100.0 | aligned | toxicity_review | CBC/symptom toxicity rule |
| COMPV4-BRCA-0020 | 1 | 1.0 | 50.56 | 50.513 | 100.0 | aligned | toxicity_review | CBC/symptom toxicity rule |
| COMPV4-BRCA-0029 | 0 | 0.0 | -34.21 | -33.951 | 5.617 | aligned | toxicity_review | CBC/symptom toxicity rule; low hybrid or weak MRI improvement |
| COMPV4-BRCA-0043 | 0 | 0.25 | 43.89 | 43.896 | 49.114 | conflicting | toxicity_review | CBC/symptom toxicity rule; classifier/regressor conflict; low hybrid or weak MRI improvement |
| COMPV4-BRCA-0046 | 1 | 1.0 | 86.33 | 86.11 | 100.0 | aligned | toxicity_review | CBC/symptom toxicity rule |
| COMPV4-BRCA-0067 | 1 | 1.0 | 64.46 | 64.617 | 100.0 | aligned | toxicity_review | CBC/symptom toxicity rule |
| COMPV4-BRCA-0071 | 1 | 1.0 | 90.69 | 90.565 | 100.0 | aligned | toxicity_review | CBC/symptom toxicity rule |
| COMPV4-BRCA-0076 | 1 | 1.0 | 87.67 | 87.594 | 100.0 | aligned | toxicity_review | CBC/symptom toxicity rule |
| COMPV4-BRCA-0081 | 0 | 0.0 | 89.7 | 89.671 | 35.0 | conflicting | toxicity_review | CBC/symptom toxicity rule; classifier/regressor conflict; low hybrid or weak MRI improvement |
| COMPV4-BRCA-0100 | 1 | 0.326234 | 48.55 | 48.518 | 55.687 | conflicting | toxicity_review | CBC/symptom toxicity rule; classifier/regressor conflict |
| COMPV4-BRCA-0104 | 0 | 0.0 | -29.03 | 34.0 | 29.4 | conflicting | toxicity_review | CBC/symptom toxicity rule; classifier/regressor conflict; low hybrid or weak MRI improvement |
| COMPV4-BRCA-0108 | 0 | 0.0 | 33.33 | 35.106 | 29.787 | conflicting | toxicity_review | CBC/symptom toxicity rule; classifier/regressor conflict; low hybrid or weak MRI improvement |
| COMPV4-BRCA-0110 | 0 | 0.0 | 42.84 | 42.733 | 32.457 | conflicting | toxicity_review | CBC/symptom toxicity rule; classifier/regressor conflict; low hybrid or weak MRI improvement |
| COMPV4-BRCA-0114 | 0 | 0.0 | -11.5 | -11.388 | 13.514 | aligned | toxicity_review | CBC/symptom toxicity rule; low hybrid or weak MRI improvement |
| COMPV4-BRCA-0115 | 1 | 1.0 | 51.55 | 51.493 | 100.0 | aligned | toxicity_review | CBC/symptom toxicity rule |
| COMPV4-BRCA-0124 | 0 | 0.0 | -25.21 | -25.283 | 8.651 | aligned | toxicity_review | CBC/symptom toxicity rule; low hybrid or weak MRI improvement |
| COMPV4-BRCA-0127 | 1 | 1.0 | 54.2 | 54.261 | 100.0 | aligned | toxicity_review | CBC/symptom toxicity rule |
| COMPV4-BRCA-0128 | 0 | 0.0 | 34.15 | 33.317 | 29.161 | conflicting | toxicity_review | CBC/symptom toxicity rule; classifier/regressor conflict; low hybrid or weak MRI improvement |
| COMPV4-BRCA-0131 | 1 | 1.0 | 75.42 | 74.97 | 100.0 | aligned | toxicity_review | CBC/symptom toxicity rule |

## Regression Slice Metrics

| slice | value | n | mae | rmse | r2 | status |
| --- | --- | --- | --- | --- | --- | --- |
| age_band | 65-74 | 25 | 4.823 | 14.465 | 0.879 | acceptable |
| age_band | <45 | 29 | 3.124 | 10.586 | 0.884 | acceptable |
| molecular_subtype | HER2+ | 25 | 5.84 | 15.158 | 0.856 | acceptable |
| regimen | TCHP | 25 | 5.84 | 15.158 | 0.856 | acceptable |
| stage | IIIA | 24 | 5.005 | 16.083 | 0.888 | acceptable |
| stage | IIIB | 19 | 5.745 | 14.712 | 0.752 | acceptable |
| stage | IV | 4 | 0.081 | 0.088 | 1.0 | low_support |
| hybrid_review_category | discordant_signal_review | 9 | 8.944 | 25.202 | 0.175 | needs_review |
| molecular_subtype | HR+/HER2+ | 8 | 6.617 | 17.884 | 0.823 | needs_review |
| regimen | TCHP then endocrine therapy | 8 | 6.617 | 17.884 | 0.823 | needs_review |
| age_band | 45-54 | 27 | 2.946 | 14.546 | 0.889 | strong |
| age_band | 55-64 | 24 | 0.665 | 2.169 | 0.997 | strong |
| hybrid_review_category | response_trend_review | 6 | 0.236 | 0.301 | 0.999 | strong |
| hybrid_review_category | routine_monitoring | 14 | 1.786 | 6.481 | 0.791 | strong |
| hybrid_review_category | toxicity_review | 76 | 2.628 | 10.261 | 0.936 | strong |
| molecular_subtype | HR+/HER2- | 47 | 1.882 | 11.035 | 0.932 | strong |
| molecular_subtype | triple-negative | 25 | 0.771 | 3.353 | 0.988 | strong |
| regimen | dose-dense AC then paclitaxel | 47 | 1.882 | 11.035 | 0.932 | strong |
| regimen | paclitaxel + carboplatin then AC | 25 | 0.771 | 3.353 | 0.988 | strong |
| stage | IIA | 26 | 2.738 | 12.378 | 0.861 | strong |
| stage | IIB | 32 | 0.184 | 0.32 | 1.0 | strong |

## Largest Regression Residuals

| patient_id | actual_response_score_percent | random_forest_regressor_response_score_percent | response_residual | absolute_response_residual | stage | molecular_subtype | regimen | hybrid_score | hybrid_review_category | rule_explanation | latest_mri_percent_change | max_symptom_severity | nadir_wbc | nadir_anc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COMPV4-BRCA-0154 | -38.08 | 37.483 | 75.56299999999999 | 75.56299999999999 | IIIA | HR+/HER2- | dose-dense AC then paclitaxel | 30.619 | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | 21.92 | 7.0 | 0.59 | 0.29 |
| COMPV4-BRCA-0104 | -29.03 | 34.0 | 63.03 | 63.03 | IIA | HER2+ | TCHP | 29.4 | toxicity_review | CBC/symptom toxicity rule; classifier/regressor conflict; low hybrid or weak MRI improvement | 11.29 | 9.0 | 0.61 | 0.27 |
| COMPV4-BRCA-0156 | 89.05 | 38.503 | -50.547 | 50.547 | IIIB | HR+/HER2+ | TCHP then endocrine therapy | 95.976 | toxicity_review | CBC/symptom toxicity rule | -73.33 | 9.0 | 0.59 | 0.26 |
| COMPV4-BRCA-0287 | 70.18 | 39.109 | -31.071000000000005 | 31.071000000000005 | IIIB | HER2+ | TCHP | 96.188 | toxicity_review | CBC/symptom toxicity rule | -54.04 | 8.0 | 0.61 | 0.29 |
| COMPV4-BRCA-0097 | 62.92 | 38.671 | -24.249000000000002 | 24.249000000000002 | IIIB | HER2+ | TCHP | 96.035 | routine_monitoring | no major synthetic review rule | -40.21 | 4.0 | 0.61 | 0.33 |
| COMPV4-BRCA-0246 | 88.62 | 71.881 | -16.739000000000004 | 16.739000000000004 | IIIA | triple-negative | paclitaxel + carboplatin then AC | 100.0 | toxicity_review | CBC/symptom toxicity rule | -70.34 | 8.0 | 0.58 | 0.28 |
| COMPV4-BRCA-0200 | 61.43 | 71.796 | 10.366000000000007 | 10.366000000000007 | IIIA | HER2+ | TCHP | 100.0 | toxicity_review | CBC/symptom toxicity rule | -46.79 | 9.0 | 0.6 | 0.39 |
| COMPV4-BRCA-0224 | -52.16 | -42.008 | 10.151999999999994 | 10.151999999999994 | IIIA | HER2+ | TCHP | 2.797 | toxicity_review | CBC/symptom toxicity rule; low hybrid or weak MRI improvement | 52.16 | 8.0 | 0.58 | 0.26 |
| COMPV4-BRCA-0213 | 31.2 | 33.356 | 2.1560000000000024 | 2.1560000000000024 | IIA | HR+/HER2- | dose-dense AC then paclitaxel | 29.175 | toxicity_review | CBC/symptom toxicity rule; classifier/regressor conflict; low hybrid or weak MRI improvement | -31.2 | 8.0 | 0.61 | 0.32 |
| COMPV4-BRCA-0348 | 28.94 | 30.813 | 1.8729999999999976 | 1.8729999999999976 | IIA | HR+/HER2+ | TCHP then endocrine therapy | 28.285 | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | -28.94 | 7.0 | 0.62 | 0.29 |
| COMPV4-BRCA-0108 | 33.33 | 35.106 | 1.7760000000000034 | 1.7760000000000034 | IIIB | HER2+ | TCHP | 29.787 | toxicity_review | CBC/symptom toxicity rule; classifier/regressor conflict; low hybrid or weak MRI improvement | -28.22 | 8.0 | 0.58 | 0.27 |
| COMPV4-BRCA-0214 | -38.41 | -40.061 | -1.6510000000000034 | 1.6510000000000034 | IIIA | HR+/HER2- | dose-dense AC then paclitaxel | 3.479 | toxicity_review | CBC/symptom toxicity rule; low hybrid or weak MRI improvement | 38.41 | 9.0 | 0.6 | 0.26 |
| COMPV4-BRCA-0378 | 30.0 | 31.552 | 1.5519999999999996 | 1.5519999999999996 | IIIA | HER2+ | TCHP | 28.543 | toxicity_review | CBC/symptom toxicity rule; classifier/regressor conflict; low hybrid or weak MRI improvement | -30.0 | 8.0 | 0.58 | 0.26 |
| COMPV4-BRCA-0188 | 30.3 | 31.442 | 1.1419999999999995 | 1.1419999999999995 | IIA | HR+/HER2- | dose-dense AC then paclitaxel | 28.505 | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | -30.3 | 7.0 | 0.58 | 0.26 |
| COMPV4-BRCA-0278 | 31.08 | 32.045 | 0.9650000000000034 | 0.9650000000000034 | IIIA | HR+/HER2- | dose-dense AC then paclitaxel | 28.716 | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | -31.08 | 7.0 | 0.58 | 0.28 |
| COMPV4-BRCA-0174 | -29.56 | -30.493 | -0.9329999999999998 | 0.9329999999999998 | IIB | HR+/HER2- | dose-dense AC then paclitaxel | 6.827 | toxicity_review | CBC/symptom toxicity rule; low hybrid or weak MRI improvement | 29.56 | 8.0 | 0.59 | 0.28 |
| COMPV4-BRCA-0274 | -29.51 | -30.37 | -0.8599999999999994 | 0.8599999999999994 | IIB | HR+/HER2- | dose-dense AC then paclitaxel | 6.87 | toxicity_review | CBC/symptom toxicity rule; low hybrid or weak MRI improvement | 29.51 | 9.0 | 0.59 | 0.29 |
| COMPV4-BRCA-0329 | -30.43 | -31.287 | -0.8569999999999993 | 0.8569999999999993 | IIB | HR+/HER2- | dose-dense AC then paclitaxel | 6.55 | toxicity_review | CBC/symptom toxicity rule; low hybrid or weak MRI improvement | 30.43 | 7.0 | 0.59 | 0.28 |
| COMPV4-BRCA-0128 | 34.15 | 33.317 | -0.8329999999999984 | 0.8329999999999984 | IIIB | HER2+ | TCHP | 29.161 | toxicity_review | CBC/symptom toxicity rule; classifier/regressor conflict; low hybrid or weak MRI improvement | -34.15 | 9.0 | 0.59 | 0.27 |
| COMPV4-BRCA-0364 | -29.66 | -30.453 | -0.7929999999999993 | 0.7929999999999993 | IIIA | HR+/HER2- | dose-dense AC then paclitaxel | 6.841 | toxicity_review | CBC/symptom toxicity rule; low hybrid or weak MRI improvement | 29.66 | 9.0 | 0.62 | 0.35 |
| COMPV4-BRCA-0208 | 31.27 | 32.048 | 0.7780000000000022 | 0.7780000000000022 | IIA | triple-negative | paclitaxel + carboplatin then AC | 28.717 | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | -31.27 | 7.0 | 0.59 | 0.33 |
| COMPV4-BRCA-0366 | 93.44 | 94.073 | 0.6329999999999956 | 0.6329999999999956 | IIB | HER2+ | TCHP | 100.0 | toxicity_review | CBC/symptom toxicity rule | -93.44 | 8.0 | 0.59 | 0.29 |
| COMPV4-BRCA-0119 | -30.81 | -31.414 | -0.6040000000000028 | 0.6040000000000028 | IIIA | HER2+ | TCHP | 6.505 | response_trend_review | low hybrid or weak MRI improvement | 30.81 | 7.0 | 0.62 | 0.33 |
| COMPV4-BRCA-0131 | 75.42 | 74.97 | -0.45000000000000284 | 0.45000000000000284 | IIB | HER2+ | TCHP | 100.0 | toxicity_review | CBC/symptom toxicity rule | -75.42 | 9.0 | 0.58 | 0.25 |
| COMPV4-BRCA-0384 | -36.25 | -35.8 | 0.45000000000000284 | 0.45000000000000284 | IIA | HR+/HER2- | dose-dense AC then paclitaxel | 4.97 | toxicity_review | CBC/symptom toxicity rule; low hybrid or weak MRI improvement | 36.25 | 8.0 | 0.59 | 0.26 |

## Output Files

- `test_set_predictions_detailed_csv`: `Data\complete_synthetic_training\detailed_eval\test_set_predictions_detailed.csv`
- `regression_slice_metrics_csv`: `Data\complete_synthetic_training\detailed_eval\regression_slice_metrics.csv`
- `regression_residual_review_csv`: `Data\complete_synthetic_training\detailed_eval\regression_residual_review.csv`
- `hybrid_threshold_policy_csv`: `Data\complete_synthetic_training\detailed_eval\hybrid_threshold_policy.csv`
- `hybrid_review_summary_csv`: `Data\complete_synthetic_training\detailed_eval\hybrid_review_summary.csv`
- `training_eval_summary_json`: `Data\complete_synthetic_training\detailed_eval\training_eval_summary.json`
- `markdown_report`: `Data\complete_synthetic_training\detailed_eval\training_eval_report.md`
- `html_report`: `Data\complete_synthetic_training\detailed_eval\training_eval_report.html`
