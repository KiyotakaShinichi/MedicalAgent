# Synthetic simple-baseline audit

This audit answers a narrow engineering question: do the synthetic monitoring
models outperform deliberately simple references on the same internal held-out
rows?

It reports classification accuracy, Brier score, and AUROC for a constant 0.5
predictor, post-hoc prevalence and majority references, logistic regression,
and the calibrated gradient-boosting champion. Regression is compared with
zero, the post-hoc test-set mean, ridge regression, and the random-forest
regressor champion.

The post-hoc references are descriptive lower-bound checks, not deployable
models. The canonical paired comparison remains the exact McNemar test against
logistic regression. If that test does not establish a positive difference,
the system must not claim that model complexity is proven better.

All rows are simulator-built. This artifact is not clinical validation, does
not represent real patients, and cannot support diagnosis, prognosis,
treatment, patient-benefit, or healthcare-production claims.

The audit also reports selective-risk curves at 100%, 90%, 75%, and 50%
coverage. Classification rows are ranked by distance from probability 0.5;
regression rows use ridge-versus-random-forest disagreement as an abstention
stress proxy. Neither measure is calibrated clinical uncertainty, and neither
shows that abstention is safe for patient care.
