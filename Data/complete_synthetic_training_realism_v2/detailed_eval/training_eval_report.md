# Detailed Synthetic Training Evaluation Report

Generated from the current local synthetic training artifacts.

## Claim Boundary

Synthetic-data engineering evaluation only. This report helps visualize model behavior, rule routing, and error modes. It is not clinical validation.

## Headline Results

- Test patients: 60
- Best classifier: `gradient_boosting`
- Best regressor: `random_forest_regressor`
- Classifier patient-level AUROC: `0.989`
- Classifier patient-level AUPRC: `0.992`
- Classifier patient-level Brier score: `0.07`
- Calibrated champion status: `trained`
- Calibrated validation ECE: `0.0219`
- Regressor patient-level MAE: `1.713`
- Regressor patient-level RMSE: `5.917`
- Regressor patient-level R2: `0.972`

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
| routine_monitoring | 29 | 99.732 | 1.0 | 65.99 | 0.0 | 0.0 | 0.483 |
| discordant_signal_review | 18 | 46.352 | 0.236 | 39.141 | 0.0 | 0.611 | 0.3 |
| response_trend_review | 12 | 11.077 | 0.0 | -18.351 | 0.0 | 1.0 | 0.2 |
| watch_closely | 1 | 59.822 | 0.405 | 45.777 | 0.0 | 0.0 | 0.017 |

## Error Taxonomy

| error_type | count | rate | example_patient_ids | meaning |
| --- | --- | --- | --- | --- |
| calibration_boundary_case | 10 | 0.167 | REALISM-BRCA-0058; REALISM-BRCA-0073; REALISM-BRCA-0076; REALISM-BRCA-0082; REALISM-BRCA-0117; REALISM-BRCA-0137; REALIS | Probability is close to the operating threshold; threshold changes may flip routing. |
| subtype_confusion | 7 | 0.117 | REALISM-BRCA-0008; REALISM-BRCA-0058; REALISM-BRCA-0082; REALISM-BRCA-0162; REALISM-BRCA-0205; REALISM-BRCA-0235; REALIS | HER2-related subgroup where classifier and response-regressor disagree. |
| false_negative_favorable_response | 4 | 0.067 | REALISM-BRCA-0073; REALISM-BRCA-0082; REALISM-BRCA-0240; REALISM-BRCA-0187 | Classifier missed a synthetically favorable final outcome. In medicine this is reviewed carefully because false negative |
| regimen_shift_uncertainty | 3 | 0.05 | REALISM-BRCA-0240; REALISM-BRCA-0050; REALISM-BRCA-0200 | Regimen-specific review gap for HR+/HER2+ TCHP followed by endocrine therapy. |
| response_regression_outlier | 1 | 0.017 | REALISM-BRCA-0131 | Continuous response estimate differs from the synthetic MRI-derived label by at least 20 percentage points. |
| delayed_toxicity_detection | 0 | 0.0 |  | Deterministic CBC/symptom toxicity rule triggers even though the response classifier is favorable. |
| false_positive_overoptimism | 0 | 0.0 |  | Classifier predicted favorable response for an unfavorable synthetic outcome. This can over-reassure a review workflow. |
| sparse_history_instability | 0 | 0.0 |  | Limited temporal history or missing model signal makes the patient-level estimate less stable. |

## Cost-Sensitive Threshold Evaluation

| threshold | fn_cost | fp_cost | weighted_cost | true_negative | false_positive | false_negative | true_positive | sensitivity | specificity | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | 10 | 1 | 6 | 21 | 6 | 0 | 33 | 1.0 | 0.778 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.3 | 10 | 1 | 7 | 20 | 7 | 0 | 33 | 1.0 | 0.741 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.5 | 10 | 1 | 40 | 27 | 0 | 4 | 29 | 0.879 | 1.0 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.6 | 10 | 1 | 40 | 27 | 0 | 4 | 29 | 0.879 | 1.0 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.7 | 10 | 1 | 40 | 27 | 0 | 4 | 29 | 0.879 | 1.0 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.4 | 5 | 1 | 6 | 21 | 6 | 0 | 33 | 1.0 | 0.778 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.3 | 5 | 1 | 7 | 20 | 7 | 0 | 33 | 1.0 | 0.741 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.5 | 5 | 1 | 20 | 27 | 0 | 4 | 29 | 0.879 | 1.0 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.6 | 5 | 1 | 20 | 27 | 0 | 4 | 29 | 0.879 | 1.0 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.7 | 5 | 1 | 20 | 27 | 0 | 4 | 29 | 0.879 | 1.0 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.4 | 3 | 1 | 6 | 21 | 6 | 0 | 33 | 1.0 | 0.778 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.3 | 3 | 1 | 7 | 20 | 7 | 0 | 33 | 1.0 | 0.741 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.5 | 3 | 1 | 12 | 27 | 0 | 4 | 29 | 0.879 | 1.0 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.6 | 3 | 1 | 12 | 27 | 0 | 4 | 29 | 0.879 | 1.0 | Lower weighted_cost is better for this synthetic threshold policy. |
| 0.7 | 3 | 1 | 12 | 27 | 0 | 4 | 29 | 0.879 | 1.0 | Lower weighted_cost is better for this synthetic threshold policy. |

## Example Test-Set Predictions

| patient_id | actual_label | champion_probability | actual_response_score_percent | champion_response_score_percent | hybrid_score | model_agreement | hybrid_review_category | rule_explanation | response_uncertainty_width | response_uncertainty_band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| REALISM-BRCA-0008 | 0 | 0.0 | 31.06 | 31.028 | 28.36 | conflicting | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | 4.883 | narrow |
| REALISM-BRCA-0053 | 0 | 0.0 | 32.5 | 32.0 | 28.7 | conflicting | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | 1.166 | narrow |
| REALISM-BRCA-0058 | 0 | 0.4 | 44.09 | 44.003 | 58.901 | conflicting | discordant_signal_review | classifier/regressor conflict | 0.601 | narrow |
| REALISM-BRCA-0069 | 0 | 0.0 | 59.5 | 59.464 | 35.0 | conflicting | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | 2.069 | narrow |
| REALISM-BRCA-0073 | 1 | 0.4 | 46.49 | 46.517 | 59.781 | conflicting | discordant_signal_review | classifier/regressor conflict | 5.768 | moderate |
| REALISM-BRCA-0076 | 0 | 0.4 | 40.53 | 40.606 | 57.712 | conflicting | discordant_signal_review | classifier/regressor conflict | 2.656 | narrow |
| REALISM-BRCA-0082 | 1 | 0.4 | 50.8 | 50.827 | 61.0 | conflicting | discordant_signal_review | classifier/regressor conflict | 1.112 | narrow |
| REALISM-BRCA-0117 | 0 | 0.4 | 30.51 | 30.474 | 54.166 | conflicting | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | 3.436 | narrow |
| REALISM-BRCA-0137 | 0 | 0.4 | 42.22 | 42.094 | 58.233 | conflicting | discordant_signal_review | classifier/regressor conflict | 3.547 | narrow |
| REALISM-BRCA-0162 | 0 | 0.0 | 36.74 | 36.727 | 30.354 | conflicting | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | 1.672 | narrow |
| REALISM-BRCA-0171 | 0 | 0.0 | 36.25 | 36.552 | 30.293 | conflicting | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | 0.756 | narrow |
| REALISM-BRCA-0199 | 0 | 0.355723 | 38.79 | 38.794 | 54.2 | conflicting | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | 0.796 | narrow |
| REALISM-BRCA-0205 | 0 | 0.4 | 43.06 | 43.149 | 58.602 | conflicting | discordant_signal_review | classifier/regressor conflict | 0.58 | narrow |
| REALISM-BRCA-0217 | 0 | 0.4 | 44.41 | 44.453 | 59.059 | conflicting | discordant_signal_review | classifier/regressor conflict | 0.516 | narrow |
| REALISM-BRCA-0219 | 0 | 0.040826 | 35.26 | 37.12 | 33.146 | conflicting | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | 5.163 | moderate |
| REALISM-BRCA-0226 | 0 | 0.0 | 28.68 | 29.3 | 27.755 | conflicting | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | 5.308 | moderate |
| REALISM-BRCA-0235 | 0 | 0.254807 | 28.28 | 29.616 | 44.428 | conflicting | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | 6.015 | moderate |
| REALISM-BRCA-0240 | 1 | 0.4 | 45.06 | 31.821 | 54.637 | conflicting | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | 8.11 | moderate |
| REALISM-BRCA-0004 | 0 | 0.0 | -18.89 | -18.961 | 10.864 | aligned | response_trend_review | low hybrid or weak MRI improvement | 7.658 | moderate |
| REALISM-BRCA-0031 | 0 | 0.0 | -11.63 | -11.34 | 13.531 | aligned | response_trend_review | low hybrid or weak MRI improvement | 6.237 | moderate |

## Regression Slice Metrics

| slice | value | n | mae | rmse | r2 | status |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid_review_category | response_trend_review | 12 | 3.626 | 10.23 | 0.056 | acceptable |
| stage | IIA | 14 | 4.16 | 10.118 | 0.928 | acceptable |
| age_band | 65-74 | 1 | 0.013 | 0.013 |  | low_support |
| hybrid_review_category | watch_closely | 1 | 0.263 | 0.263 |  | low_support |
| stage | IV | 1 | 0.036 | 0.036 |  | low_support |
| age_band | 45-54 | 31 | 2.261 | 7.431 | 0.941 | strong |
| age_band | 55-64 | 8 | 0.34 | 0.674 | 0.994 | strong |
| age_band | <45 | 20 | 1.499 | 4.388 | 0.99 | strong |
| hybrid_review_category | discordant_signal_review | 18 | 1.025 | 3.174 | 0.843 | strong |
| hybrid_review_category | routine_monitoring | 29 | 1.399 | 4.783 | 0.881 | strong |
| molecular_subtype | HER2+ | 15 | 1.389 | 4.578 | 0.976 | strong |
| molecular_subtype | HR+/HER2+ | 5 | 2.793 | 5.925 | 0.975 | strong |
| molecular_subtype | HR+/HER2- | 27 | 2.183 | 7.644 | 0.944 | strong |
| molecular_subtype | triple-negative | 13 | 0.697 | 1.597 | 0.999 | strong |
| regimen | TCHP | 15 | 1.389 | 4.578 | 0.976 | strong |
| regimen | TCHP then endocrine therapy | 5 | 2.793 | 5.925 | 0.975 | strong |
| regimen | dose-dense AC then paclitaxel | 27 | 2.183 | 7.644 | 0.944 | strong |
| regimen | paclitaxel + carboplatin then AC | 13 | 0.697 | 1.597 | 0.999 | strong |
| stage | IIB | 23 | 0.221 | 0.424 | 1.0 | strong |
| stage | IIIA | 13 | 2.945 | 7.141 | 0.965 | strong |
| stage | IIIB | 9 | 0.128 | 0.182 | 1.0 | strong |

## Largest Regression Residuals

| patient_id | actual_response_score_percent | random_forest_regressor_response_score_percent | response_residual | absolute_response_residual | stage | molecular_subtype | regimen | hybrid_score | hybrid_review_category | rule_explanation | response_uncertainty_lower | response_uncertainty_upper | response_uncertainty_width | response_uncertainty_band | latest_mri_percent_change | max_symptom_severity | nadir_wbc | nadir_anc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| REALISM-BRCA-0131 | -16.62 | 18.352 | 34.972 | 34.972 | IIA | HR+/HER2- | dose-dense AC then paclitaxel | 23.923 | response_trend_review | low hybrid or weak MRI improvement | 14.261 | 27.586 | 13.325 | moderate | 6.34 | 6.0 | 7.09 | 3.39 |
| REALISM-BRCA-0057 | 61.3 | 42.585 | -18.714999999999996 | 18.714999999999996 | IIIA | HR+/HER2- | dose-dense AC then paclitaxel | 97.405 | routine_monitoring | no major synthetic review rule | 25.894 | 43.341 | 17.447 | wide | -50.43 | 7.0 | 2.42 | 1.59 |
| REALISM-BRCA-0028 | 60.71 | 43.043 | -17.667 | 17.667 | IIIA | HER2+ | TCHP | 97.565 | routine_monitoring | no major synthetic review rule | 25.893 | 43.734 | 17.841 | wide | -39.52 | 7.0 | 3.18 | 1.48 |
| REALISM-BRCA-0240 | 45.06 | 31.821 | -13.239 | 13.239 | IIA | HR+/HER2+ | TCHP then endocrine therapy | 54.637 | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | 23.531 | 31.641 | 8.11 | moderate | -36.05 | 7.0 | 1.84 | 1.61 |
| REALISM-BRCA-0086 | -47.06 | -41.476 | 5.584000000000003 | 5.584000000000003 | IIA | triple-negative | paclitaxel + carboplatin then AC | 2.983 | response_trend_review | low hybrid or weak MRI improvement | -45.79 | -33.002 | 12.788 | moderate | 47.06 | 7.0 | 3.3 | 1.63 |
| REALISM-BRCA-0219 | 35.26 | 37.12 | 1.8599999999999994 | 1.8599999999999994 | IIB | HR+/HER2- | dose-dense AC then paclitaxel | 33.146 | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | 35.271 | 40.434 | 5.163 | moderate | -35.26 | 7.0 | 2.92 | 1.61 |
| REALISM-BRCA-0235 | 28.28 | 29.616 | 1.3359999999999985 | 1.3359999999999985 | IIA | HER2+ | TCHP | 44.428 | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | 28.616 | 34.631 | 6.015 | moderate | -28.28 | 5.0 | 2.97 | 1.9 |
| REALISM-BRCA-0136 | -26.19 | -27.162 | -0.9719999999999978 | 0.9719999999999978 | IIA | triple-negative | paclitaxel + carboplatin then AC | 7.993 | response_trend_review | low hybrid or weak MRI improvement | -26.934 | -12.467 | 14.467 | moderate | 26.19 | 7.0 | 3.02 | 1.46 |
| REALISM-BRCA-0226 | 28.68 | 29.3 | 0.620000000000001 | 0.620000000000001 | IIA | triple-negative | paclitaxel + carboplatin then AC | 27.755 | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | 29.114 | 34.422 | 5.308 | moderate | -28.68 | 7.0 | 5.68 | 3.19 |
| REALISM-BRCA-0056 | 92.66 | 92.147 | -0.512999999999991 | 0.512999999999991 | IIA | triple-negative | paclitaxel + carboplatin then AC | 100.0 | routine_monitoring | no major synthetic review rule | 85.169 | 92.506 | 7.337 | moderate | -92.66 | 6.0 | 3.59 | 1.91 |
| REALISM-BRCA-0053 | 32.5 | 32.0 | -0.5 | 0.5 | IIA | triple-negative | paclitaxel + carboplatin then AC | 28.7 | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | 32.121 | 33.287 | 1.166 | narrow | -32.5 | 6.0 | 4.23 | 2.06 |
| REALISM-BRCA-0224 | 84.47 | 84.002 | -0.4680000000000035 | 0.4680000000000035 | IIIA | HR+/HER2- | dose-dense AC then paclitaxel | 100.0 | routine_monitoring | no major synthetic review rule | 81.869 | 84.449 | 2.58 | narrow | -84.47 | 7.0 | 4.76 | 2.27 |
| REALISM-BRCA-0050 | -27.87 | -28.269 | -0.39899999999999736 | 0.39899999999999736 | IIIB | HR+/HER2+ | TCHP then endocrine therapy | 7.606 | response_trend_review | low hybrid or weak MRI improvement | -27.99 | -20.456 | 7.534 | moderate | 27.87 | 7.0 | 4.96 | 3.07 |
| REALISM-BRCA-0001 | 91.26 | 91.625 | 0.3649999999999949 | 0.3649999999999949 | IIIA | HR+/HER2- | dose-dense AC then paclitaxel | 100.0 | routine_monitoring | no major synthetic review rule | 84.176 | 91.528 | 7.352 | moderate | -91.26 | 6.0 | 3.63 | 1.78 |
| REALISM-BRCA-0118 | -2.68 | -3.042 | -0.36199999999999966 | 0.36199999999999966 | IIIA | HER2+ | TCHP | 16.435 | response_trend_review | low hybrid or weak MRI improvement | -4.928 | 3.54 | 8.468 | moderate | 2.68 | 7.0 | 5.03 | 2.76 |
| REALISM-BRCA-0168 | -21.35 | -21.665 | -0.3149999999999977 | 0.3149999999999977 | IIIA | HER2+ | TCHP | 9.917 | response_trend_review | low hybrid or weak MRI improvement | -22.088 | -8.277 | 13.811 | moderate | 21.35 | 6.0 | 6.49 | 3.11 |
| REALISM-BRCA-0171 | 36.25 | 36.552 | 0.3019999999999996 | 0.3019999999999996 | IIB | HR+/HER2- | dose-dense AC then paclitaxel | 30.293 | discordant_signal_review | classifier/regressor conflict; low hybrid or weak MRI improvement | 36.05 | 36.806 | 0.756 | narrow | -36.25 | 3.0 | 5.04 | 3.32 |
| REALISM-BRCA-0157 | 73.7 | 73.402 | -0.2980000000000018 | 0.2980000000000018 | IIB | HR+/HER2- | dose-dense AC then paclitaxel | 100.0 | routine_monitoring | no major synthetic review rule | 72.474 | 73.882 | 1.408 | narrow | -73.7 | 6.0 | 5.4 | 3.22 |
| REALISM-BRCA-0031 | -11.63 | -11.34 | 0.2900000000000009 | 0.2900000000000009 | IIB | HR+/HER2- | dose-dense AC then paclitaxel | 13.531 | response_trend_review | low hybrid or weak MRI improvement | -12.115 | -5.878 | 6.237 | moderate | 11.63 | 6.0 | 6.28 | 3.08 |
| REALISM-BRCA-0238 | 83.16 | 82.892 | -0.2680000000000007 | 0.2680000000000007 | IIB | HER2+ | TCHP | 100.0 | routine_monitoring | no major synthetic review rule | 77.282 | 83.708 | 6.426 | moderate | -83.16 | 7.0 | 3.14 | 1.97 |
| REALISM-BRCA-0187 | 46.04 | 45.777 | -0.2629999999999981 | 0.2629999999999981 | IIIB | HR+/HER2- | dose-dense AC then paclitaxel | 59.822 | watch_closely | no major synthetic review rule | 45.58 | 47.617 | 2.037 | narrow | -46.04 | 7.0 | 3.29 | 1.98 |
| REALISM-BRCA-0060 | 46.0 | 45.746 | -0.2539999999999978 | 0.2539999999999978 | IIB | HR+/HER2+ | TCHP then endocrine therapy | 98.511 | routine_monitoring | no major synthetic review rule | 45.824 | 49.309 | 3.485 | narrow | -46.0 | 7.0 | 4.8 | 2.92 |
| REALISM-BRCA-0111 | 64.92 | 65.163 | 0.242999999999995 | 0.242999999999995 | IIB | HR+/HER2- | dose-dense AC then paclitaxel | 100.0 | routine_monitoring | no major synthetic review rule | 63.957 | 65.511 | 1.554 | narrow | -64.92 | 7.0 | 4.66 | 2.76 |
| REALISM-BRCA-0046 | 46.2 | 46.413 | 0.21299999999999386 | 0.21299999999999386 | IIB | triple-negative | paclitaxel + carboplatin then AC | 98.745 | routine_monitoring | no major synthetic review rule | 46.241 | 49.715 | 3.474 | narrow | -46.2 | 7.0 | 4.54 | 2.57 |
| REALISM-BRCA-0186 | -25.79 | -25.587 | 0.2029999999999994 | 0.2029999999999994 | IIB | triple-negative | paclitaxel + carboplatin then AC | 8.545 | response_trend_review | low hybrid or weak MRI improvement | -25.673 | -13.622 | 12.051 | moderate | 25.79 | 7.0 | 2.57 | 1.47 |

## Output Files

- `test_set_predictions_detailed_csv`: `Data\complete_synthetic_training_realism_v2\detailed_eval\test_set_predictions_detailed.csv`
- `regression_slice_metrics_csv`: `Data\complete_synthetic_training_realism_v2\detailed_eval\regression_slice_metrics.csv`
- `regression_residual_review_csv`: `Data\complete_synthetic_training_realism_v2\detailed_eval\regression_residual_review.csv`
- `hybrid_threshold_policy_csv`: `Data\complete_synthetic_training_realism_v2\detailed_eval\hybrid_threshold_policy.csv`
- `hybrid_review_summary_csv`: `Data\complete_synthetic_training_realism_v2\detailed_eval\hybrid_review_summary.csv`
- `error_taxonomy_csv`: `Data\complete_synthetic_training_realism_v2\detailed_eval\error_taxonomy.csv`
- `cost_sensitive_evaluation_csv`: `Data\complete_synthetic_training_realism_v2\detailed_eval\cost_sensitive_evaluation.csv`
- `training_eval_summary_json`: `Data\complete_synthetic_training_realism_v2\detailed_eval\training_eval_summary.json`
- `markdown_report`: `Data\complete_synthetic_training_realism_v2\detailed_eval\training_eval_report.md`
- `html_report`: `Data\complete_synthetic_training_realism_v2\detailed_eval\training_eval_report.html`
