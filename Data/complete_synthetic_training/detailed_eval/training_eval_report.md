# Detailed Synthetic Training Evaluation Report

Generated from the current local synthetic training artifacts.

## Claim Boundary

Synthetic-data engineering evaluation only. This report helps visualize model behavior, rule routing, and error modes. It is not clinical validation.

## Headline Results

- Test patients: 120
- Best classifier: `gradient_boosting`
- Best regressor: `random_forest_regressor`
- Classifier patient-level AUROC: `0.995`
- Classifier patient-level AUPRC: `0.996`
- Classifier patient-level Brier score: `0.047`
- Calibrated champion status: `trained`
- Calibrated validation ECE: `0.0597`
- Regressor patient-level MAE: `1.48`
- Regressor patient-level RMSE: `6.479`
- Regressor patient-level R2: `0.971`

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
| toxicity_review | 87 | 68.674 | 0.584 | 51.234 | 1.0 | 0.391 | 0.725 |
| routine_monitoring | 11 | 98.143 | 0.974 | 70.249 | 0.0 | 0.0 | 0.092 |
| response_trend_review | 11 | 8.79 | 0.0 | -24.885 | 0.0 | 1.0 | 0.092 |
| discordant_signal_review | 10 | 32.454 | 0.026 | 41.296 | 0.0 | 1.0 | 0.083 |
| watch_closely | 1 | 74.612 | 0.624 | 47.29 | 0.0 | 0.0 | 0.008 |

## Error Taxonomy

| error_type | count | rate | example_patient_ids | meaning |
| --- | --- | --- | --- | --- |
| delayed_toxicity_detection | 52 | 0.433 | COMPV5-BRCA-0005; COMPV5-BRCA-0014; COMPV5-BRCA-0079; COMPV5-BRCA-0088; COMPV5-BRCA-0092; COMPV5-BRCA-0102; COMPV5-BRCA- | Deterministic CBC/symptom toxicity rule triggers even though the response classifier is favorable. |
| subtype_confusion | 12 | 0.1 | COMPV5-BRCA-0112; COMPV5-BRCA-0176; COMPV5-BRCA-0244; COMPV5-BRCA-0326; COMPV5-BRCA-0444; COMPV5-BRCA-0508; COMPV5-BRCA- | HER2-related subgroup where classifier and response-regressor disagree. |
| regimen_shift_uncertainty | 4 | 0.033 | COMPV5-BRCA-0144; COMPV5-BRCA-0594; COMPV5-BRCA-0154; COMPV5-BRCA-0354 | Regimen-specific review gap for HR+/HER2+ TCHP followed by endocrine therapy. |
| false_negative_favorable_response | 3 | 0.025 | COMPV5-BRCA-0555; COMPV5-BRCA-0587; COMPV5-BRCA-0369 | Classifier missed a synthetically favorable final outcome. In medicine this is reviewed carefully because false negative |
| response_regression_outlier | 3 | 0.025 | COMPV5-BRCA-0277; COMPV5-BRCA-0337; COMPV5-BRCA-0507 | Continuous response estimate differs from the synthetic MRI-derived label by at least 20 percentage points. |
| false_positive_overoptimism | 1 | 0.008 | COMPV5-BRCA-0346 | Classifier predicted favorable response for an unfavorable synthetic outcome. This can over-reassure a review workflow. |
| calibration_boundary_case | 0 | 0.0 |  | Probability is close to the operating threshold; threshold changes may flip routing. |
| sparse_history_instability | 0 | 0.0 |  | Limited temporal history or missing model signal makes the patient-level estimate less stable. |

## Cost-Sensitive Threshold Evaluation

| threshold | fn_cost | fp_cost | weighted_cost | true_negative | false_positive | false_negative | true_positive | sensitivity | specificity | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3 | 10 | 1 | 21 | 53 | 1 | 2 | 64 | 0.97 | 0.981 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.4 | 10 | 1 | 31 | 53 | 1 | 3 | 63 | 0.955 | 0.981 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.5 | 10 | 1 | 31 | 53 | 1 | 3 | 63 | 0.955 | 0.981 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.6 | 10 | 1 | 31 | 53 | 1 | 3 | 63 | 0.955 | 0.981 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.7 | 10 | 1 | 41 | 53 | 1 | 4 | 62 | 0.939 | 0.981 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.3 | 5 | 1 | 11 | 53 | 1 | 2 | 64 | 0.97 | 0.981 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.4 | 5 | 1 | 16 | 53 | 1 | 3 | 63 | 0.955 | 0.981 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.5 | 5 | 1 | 16 | 53 | 1 | 3 | 63 | 0.955 | 0.981 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.6 | 5 | 1 | 16 | 53 | 1 | 3 | 63 | 0.955 | 0.981 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.7 | 5 | 1 | 21 | 53 | 1 | 4 | 62 | 0.939 | 0.981 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.3 | 3 | 1 | 7 | 53 | 1 | 2 | 64 | 0.97 | 0.981 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.4 | 3 | 1 | 10 | 53 | 1 | 3 | 63 | 0.955 | 0.981 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.5 | 3 | 1 | 10 | 53 | 1 | 3 | 63 | 0.955 | 0.981 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.6 | 3 | 1 | 10 | 53 | 1 | 3 | 63 | 0.955 | 0.981 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.7 | 3 | 1 | 13 | 53 | 1 | 4 | 62 | 0.939 | 0.981 | Lower weighted_cost is better for this synthetic threshold policy. |

## Example Test-Set Predictions

| patient_id | actual_label | champion_probability | actual_response_score_percent | champion_response_score_percent | hybrid_score | model_agreement | hybrid_review_category | rule_explanation | response_uncertainty_width | response_uncertainty_band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COMPV5-BRCA-0004 | 0 | 0.0 | -42.5 | -36.032 | 4.889 | aligned | toxicity_review | CBC/symptom toxicity rule; low hybrid or weak MRI improvement | 12.294 | moderate |
| COMPV5-BRCA-0005 | 1 | 0.857143 | 58.6 | 58.66 | 90.714 | aligned | toxicity_review | CBC/symptom toxicity rule | 2.637 | narrow |
| COMPV5-BRCA-0014 | 1 | 0.857143 | 50.0 | 50.011 | 90.714 | aligned | toxicity_review | CBC/symptom toxicity rule | 1.344 | narrow |
| COMPV5-BRCA-0022 | 0 | 0.0 | -6.46 | -6.372 | 15.27 | partially_aligned | toxicity_review | CBC/symptom toxicity rule; low hybrid or weak MRI improvement | 3.162 | narrow |
| COMPV5-BRCA-0068 | 0 | 0.0 | -20.97 | -21.0 | 10.15 | aligned | toxicity_review | CBC/symptom toxicity rule; low hybrid or weak MRI improvement | 3.876 | narrow |
| COMPV5-BRCA-0079 | 1 | 1.0 | 82.76 | 82.903 | 100.0 | aligned | toxicity_review | CBC/symptom toxicity rule | 1.136 | narrow |
| COMPV5-BRCA-0088 | 1 | 1.0 | 92.43 | 92.862 | 100.0 | aligned | toxicity_review | CBC/symptom toxicity rule | 4.958 | narrow |
| COMPV5-BRCA-0092 | 1 | 1.0 | 90.47 | 90.351 | 100.0 | aligned | toxicity_review | CBC/symptom toxicity rule | 5.547 | moderate |
| COMPV5-BRCA-0102 | 1 | 1.0 | 72.27 | 72.136 | 100.0 | aligned | toxicity_review | CBC/symptom toxicity rule | 2.46 | narrow |
| COMPV5-BRCA-0112 | 0 | 0.0 | 36.35 | 26.256 | 26.69 | conflicting | toxicity_review | CBC/symptom toxicity rule; classifier/regressor conflict; low hybrid or weak MRI improvement | 4.823 | narrow |
| COMPV5-BRCA-0115 | 1 | 1.0 | 86.1 | 86.04 | 100.0 | aligned | toxicity_review | CBC/symptom toxicity rule | 1.24 | narrow |
| COMPV5-BRCA-0119 | 1 | 1.0 | 59.28 | 59.224 | 100.0 | aligned | toxicity_review | CBC/symptom toxicity rule | 0.423 | narrow |
| COMPV5-BRCA-0133 | 1 | 1.0 | 88.06 | 87.627 | 100.0 | aligned | toxicity_review | CBC/symptom toxicity rule | 2.544 | narrow |
| COMPV5-BRCA-0137 | 0 | 0.0 | 43.75 | 43.727 | 32.804 | conflicting | toxicity_review | CBC/symptom toxicity rule; classifier/regressor conflict; low hybrid or weak MRI improvement | 1.843 | narrow |
| COMPV5-BRCA-0149 | 0 | 0.0 | 39.85 | 39.681 | 31.388 | conflicting | toxicity_review | CBC/symptom toxicity rule; classifier/regressor conflict; low hybrid or weak MRI improvement | 2.018 | narrow |
| COMPV5-BRCA-0165 | 0 | 0.0 | 88.59 | 79.558 | 35.0 | conflicting | toxicity_review | CBC/symptom toxicity rule; classifier/regressor conflict; low hybrid or weak MRI improvement | 54.042 | wide |
| COMPV5-BRCA-0167 | 0 | 0.0 | 34.47 | 34.347 | 29.521 | conflicting | toxicity_review | CBC/symptom toxicity rule; classifier/regressor conflict; low hybrid or weak MRI improvement | 0.749 | narrow |
| COMPV5-BRCA-0176 | 0 | 0.0 | 32.26 | 32.145 | 28.751 | conflicting | toxicity_review | CBC/symptom toxicity rule; classifier/regressor conflict; low hybrid or weak MRI improvement | 0.483 | narrow |
| COMPV5-BRCA-0193 | 1 | 1.0 | 55.78 | 55.783 | 100.0 | aligned | toxicity_review | CBC/symptom toxicity rule | 1.595 | narrow |
| COMPV5-BRCA-0199 | 0 | 0.0 | 41.03 | 41.026 | 31.859 | conflicting | toxicity_review | CBC/symptom toxicity rule; classifier/regressor conflict; low hybrid or weak MRI improvement | 1.821 | narrow |

## Regression Slice Metrics

| slice | value | n | mae | rmse | r2 | status |
| --- | --- | --- | --- | --- | --- | --- |
| molecular_subtype | triple-negative | 28 | 4.265 | 12.886 | 0.916 | acceptable |
| regimen | paclitaxel + carboplatin then AC | 28 | 4.265 | 12.886 | 0.916 | acceptable |
| stage | IV | 10 | 4.349 | 9.629 | 0.896 | acceptable |
| hybrid_review_category | watch_closely | 1 | 0.0 | 0.0 |  | low_support |
| age_band | 45-54 | 27 | 0.528 | 1.962 | 0.996 | strong |
| age_band | 55-64 | 27 | 2.935 | 11.187 | 0.92 | strong |
| age_band | 65-74 | 27 | 1.546 | 5.022 | 0.984 | strong |
| age_band | <45 | 39 | 1.085 | 4.731 | 0.985 | strong |
| hybrid_review_category | discordant_signal_review | 10 | 1.262 | 3.344 | 0.947 | strong |
| hybrid_review_category | response_trend_review | 11 | 0.402 | 0.713 | 0.993 | strong |
| hybrid_review_category | routine_monitoring | 11 | 0.137 | 0.168 | 1.0 | strong |
| hybrid_review_category | toxicity_review | 87 | 1.828 | 7.519 | 0.953 | strong |
| molecular_subtype | HER2+ | 23 | 0.733 | 2.336 | 0.995 | strong |
| molecular_subtype | HR+/HER2+ | 22 | 0.705 | 1.742 | 0.998 | strong |
| molecular_subtype | HR+/HER2- | 47 | 0.549 | 2.038 | 0.997 | strong |
| regimen | TCHP | 23 | 0.733 | 2.336 | 0.995 | strong |
| regimen | TCHP then endocrine therapy | 22 | 0.705 | 1.742 | 0.998 | strong |
| regimen | dose-dense AC then paclitaxel | 47 | 0.549 | 2.038 | 0.997 | strong |
| stage | IIA | 23 | 2.664 | 11.778 | 0.915 | strong |
| stage | IIB | 32 | 0.955 | 2.828 | 0.994 | strong |
| stage | IIIA | 33 | 0.229 | 0.847 | 1.0 | strong |
| stage | IIIB | 22 | 1.578 | 5.392 | 0.982 | strong |

## Largest Regression Residuals

| patient_id | actual_response_score_percent | random_forest_regressor_response_score_percent | response_residual | absolute_response_residual | stage | molecular_subtype | regimen | hybrid_score | hybrid_review_category | rule_explanation | response_uncertainty_lower | response_uncertainty_upper | response_uncertainty_width | response_uncertainty_band | latest_mri_percent_change | max_symptom_severity | nadir_wbc | nadir_anc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COMPV5-BRCA-0277 | -28.0 | 28.441 | 56.441 | 56.441 | IIA | triple-negative | paclitaxel + carboplatin then AC | 27.454 | toxicity_review | CBC/symptom toxicity rule; classifier/regressor conflict; low hybrid or weak MRI improvement | 25.169 | 28.627 | 3.458 | narrow | 14.4 | 7.0 | 0.61 | 0.32 |
| COMPV5-BRCA-0507 | 62.59 | 33.864 | -28.726000000000006 | 28.726000000000006 | IV | triple-negative | paclitaxel + carboplatin then AC | 29.352 | toxicity_review | CBC/symptom toxicity rule; classifier/regressor conflict; low hybrid or weak MRI improvement | 26.835 | 34.5 | 7.665 | moderate | -54.81 | 7.0 | 0.59 | 0.26 |
| COMPV5-BRCA-0337 | 56.95 | 32.405 | -24.545 | 24.545 | IIIB | triple-negative | paclitaxel + carboplatin then AC | 84.556 | toxicity_review | CBC/symptom toxicity rule | 25.655 | 32.693 | 7.038 | moderate | -47.12 | 8.0 | 0.59 | 0.32 |
| COMPV5-BRCA-0549 | 33.92 | 23.393 | -10.527000000000001 | 10.527000000000001 | IIB | HR+/HER2- | dose-dense AC then paclitaxel | 25.688 | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | 22.545 | 27.785 | 5.24 | moderate | -30.0 | 7.0 | 0.6 | 0.33 |
| COMPV5-BRCA-0112 | 36.35 | 26.256 | -10.094000000000001 | 10.094000000000001 | IIB | HER2+ | TCHP | 26.69 | toxicity_review | CBC/symptom toxicity rule; classifier/regressor conflict; low hybrid or weak MRI improvement | 21.907 | 26.73 | 4.823 | narrow | -32.06 | 8.0 | 0.6 | 0.31 |
| COMPV5-BRCA-0165 | 88.59 | 79.558 | -9.031999999999996 | 9.031999999999996 | IV | HR+/HER2- | dose-dense AC then paclitaxel | 35.0 | toxicity_review | CBC/symptom toxicity rule; classifier/regressor conflict; low hybrid or weak MRI improvement | 28.788 | 82.83 | 54.042 | wide | -75.94 | 9.0 | 0.58 | 0.3 |
| COMPV5-BRCA-0004 | -42.5 | -36.032 | 6.4680000000000035 | 6.4680000000000035 | IIB | HR+/HER2+ | TCHP then endocrine therapy | 4.889 | toxicity_review | CBC/symptom toxicity rule; low hybrid or weak MRI improvement | -41.003 | -28.709 | 12.294 | moderate | 42.5 | 6.0 | 0.58 | 0.27 |
| COMPV5-BRCA-0527 | -41.79 | -35.889 | 5.900999999999996 | 5.900999999999996 | IIIB | triple-negative | paclitaxel + carboplatin then AC | 4.939 | toxicity_review | CBC/symptom toxicity rule; low hybrid or weak MRI improvement | -42.474 | -27.622 | 14.852 | moderate | 41.79 | 8.0 | 0.59 | 0.32 |
| COMPV5-BRCA-0206 | 85.42 | 80.604 | -4.8160000000000025 | 4.8160000000000025 | IIIA | HER2+ | TCHP | 100.0 | toxicity_review | CBC/symptom toxicity rule | 28.697 | 84.178 | 55.481 | wide | -55.0 | 8.0 | 0.58 | 0.28 |
| COMPV5-BRCA-0244 | 32.86 | 28.388 | -4.471999999999998 | 4.471999999999998 | IV | HR+/HER2+ | TCHP then endocrine therapy | 27.436 | toxicity_review | CBC/symptom toxicity rule; classifier/regressor conflict; low hybrid or weak MRI improvement | 23.437 | 28.034 | 4.597 | narrow | -22.5 | 9.0 | 0.61 | 0.27 |
| COMPV5-BRCA-0354 | -37.95 | -36.008 | 1.9420000000000002 | 1.9420000000000002 | IIA | HR+/HER2+ | TCHP then endocrine therapy | 4.897 | response_trend_review | low hybrid or weak MRI improvement | -37.489 | -27.709 | 9.78 | moderate | 37.95 | 7.0 | 1.86 | 0.8 |
| COMPV5-BRCA-0531 | -33.94 | -34.893 | -0.953000000000003 | 0.953000000000003 | IIIB | HR+/HER2- | dose-dense AC then paclitaxel | 5.287 | response_trend_review | low hybrid or weak MRI improvement | -34.531 | -29.382 | 5.149 | moderate | 33.94 | 7.0 | 0.6 | 0.38 |
| COMPV5-BRCA-0045 | -36.54 | -35.655 | 0.884999999999998 | 0.884999999999998 | IIB | HR+/HER2- | dose-dense AC then paclitaxel | 5.021 | response_trend_review | low hybrid or weak MRI improvement | -36.193 | -26.808 | 9.385 | moderate | 36.54 | 7.0 | 0.58 | 0.31 |
| COMPV5-BRCA-0288 | 96.55 | 95.753 | -0.796999999999997 | 0.796999999999997 | IIIB | HR+/HER2+ | TCHP then endocrine therapy | 100.0 | toxicity_review | CBC/symptom toxicity rule | 90.401 | 96.641 | 6.24 | moderate | -96.55 | 9.0 | 0.59 | 0.29 |
| COMPV5-BRCA-0040 | 34.09 | 33.336 | -0.7540000000000049 | 0.7540000000000049 | IIA | HR+/HER2- | dose-dense AC then paclitaxel | 29.168 | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | 30.519 | 33.937 | 3.418 | narrow | -34.09 | 6.0 | 0.59 | 0.34 |
| COMPV5-BRCA-0313 | -24.17 | -23.69 | 0.4800000000000004 | 0.4800000000000004 | IIB | triple-negative | paclitaxel + carboplatin then AC | 9.208 | toxicity_review | CBC/symptom toxicity rule; low hybrid or weak MRI improvement | -24.107 | -18.54 | 5.567 | moderate | 24.17 | 7.0 | 0.59 | 0.31 |
| COMPV5-BRCA-0594 | 33.33 | 32.886 | -0.4439999999999955 | 0.4439999999999955 | IIIB | HR+/HER2+ | TCHP then endocrine therapy | 29.01 | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | 33.197 | 34.105 | 0.908 | narrow | -33.33 | 9.0 | 1.45 | 0.73 |
| COMPV5-BRCA-0062 | 31.67 | 32.104 | 0.4339999999999975 | 0.4339999999999975 | IV | HER2+ | TCHP | 28.736 | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | 31.974 | 35.105 | 3.131 | narrow | -31.67 | 6.0 | 0.59 | 0.36 |
| COMPV5-BRCA-0133 | 88.06 | 87.627 | -0.43300000000000693 | 0.43300000000000693 | IIB | triple-negative | paclitaxel + carboplatin then AC | 100.0 | toxicity_review | CBC/symptom toxicity rule | 85.713 | 88.257 | 2.544 | narrow | -88.06 | 8.0 | 0.59 | 0.3 |
| COMPV5-BRCA-0088 | 92.43 | 92.862 | 0.43199999999998795 | 0.43199999999998795 | IIIB | HR+/HER2+ | TCHP then endocrine therapy | 100.0 | toxicity_review | CBC/symptom toxicity rule | 87.838 | 92.796 | 4.958 | narrow | -92.43 | 8.0 | 0.61 | 0.28 |
| COMPV5-BRCA-0497 | 88.0 | 87.575 | -0.42499999999999716 | 0.42499999999999716 | IV | triple-negative | paclitaxel + carboplatin then AC | 35.0 | toxicity_review | CBC/symptom toxicity rule; classifier/regressor conflict; low hybrid or weak MRI improvement | 83.939 | 88.075 | 4.136 | narrow | -88.0 | 8.0 | 0.58 | 0.31 |
| COMPV5-BRCA-0597 | 86.35 | 85.972 | -0.3780000000000001 | 0.3780000000000001 | IIIB | triple-negative | paclitaxel + carboplatin then AC | 100.0 | toxicity_review | CBC/symptom toxicity rule | 80.978 | 86.418 | 5.44 | moderate | -86.35 | 9.0 | 0.6 | 0.27 |
| COMPV5-BRCA-0477 | -17.93 | -17.583 | 0.3470000000000013 | 0.3470000000000013 | IIA | triple-negative | paclitaxel + carboplatin then AC | 11.346 | toxicity_review | CBC/symptom toxicity rule; low hybrid or weak MRI improvement | -17.968 | -10.074 | 7.894 | moderate | 17.93 | 9.0 | 0.6 | 0.42 |
| COMPV5-BRCA-0413 | -23.95 | -23.613 | 0.33699999999999974 | 0.33699999999999974 | IIIA | triple-negative | paclitaxel + carboplatin then AC | 9.235 | response_trend_review | low hybrid or weak MRI improvement | -24.536 | -17.815 | 6.721 | moderate | 23.95 | 6.0 | 0.94 | 0.66 |
| COMPV5-BRCA-0029 | 89.0 | 89.305 | 0.3050000000000068 | 0.3050000000000068 | IIA | HR+/HER2- | dose-dense AC then paclitaxel | 100.0 | routine_monitoring | no major synthetic review rule | 81.668 | 89.46 | 7.792 | moderate | -89.0 | 6.0 | 0.61 | 0.35 |

## Output Files

- `test_set_predictions_detailed_csv`: `Data\complete_synthetic_training\detailed_eval\test_set_predictions_detailed.csv`
- `regression_slice_metrics_csv`: `Data\complete_synthetic_training\detailed_eval\regression_slice_metrics.csv`
- `regression_residual_review_csv`: `Data\complete_synthetic_training\detailed_eval\regression_residual_review.csv`
- `hybrid_threshold_policy_csv`: `Data\complete_synthetic_training\detailed_eval\hybrid_threshold_policy.csv`
- `hybrid_review_summary_csv`: `Data\complete_synthetic_training\detailed_eval\hybrid_review_summary.csv`
- `error_taxonomy_csv`: `Data\complete_synthetic_training\detailed_eval\error_taxonomy.csv`
- `cost_sensitive_evaluation_csv`: `Data\complete_synthetic_training\detailed_eval\cost_sensitive_evaluation.csv`
- `training_eval_summary_json`: `Data\complete_synthetic_training\detailed_eval\training_eval_summary.json`
- `markdown_report`: `Data\complete_synthetic_training\detailed_eval\training_eval_report.md`
- `html_report`: `Data\complete_synthetic_training\detailed_eval\training_eval_report.html`
