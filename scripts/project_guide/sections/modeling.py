"""Sections 08-10: the synthetic ML lifecycle, the classification and
regression mathematics, and calibration/abstention.

Extracted verbatim from ``build_story`` in
``scripts/generate_project_guide_pdf.py``, which had grown to 1032 lines in a
single function. Flowable content and ordering are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from reportlab.lib.units import mm
from scripts.project_guide.components import P, bullets, callout, data_table, flow_diagram, metric_row, page_break, section
from scripts.project_guide.evidence import _fmt

if TYPE_CHECKING:
    from scripts.project_guide.evidence import Evidence

def build(story: list[Any], ev: Evidence) -> None:
    """Append this module's sections to `story`, in order."""
    per_head = ev.per_head
    conformal = ev.conformal

    section(story, "08", "Synthetic ML and MLE lifecycle", "The ML layer demonstrates lifecycle discipline; it does not demonstrate clinical validity.")
    story.append(
        flow_diagram(
            [
                ("Synthetic journeys", "Versioned longitudinal generator, seeds, missingness, noise scenarios"),
                ("Contracts", "Schema, ranges, identity, date ordering, target definitions"),
                ("Features", "Patient-cycle rows, modality indicators, temporal summaries"),
                ("Splits", "Patient-grouped temporal folds; no patient overlap"),
                ("Train", "Linear, tree, kernel/MLP, temporal and small Transformer candidates"),
                ("Evaluate", "Discrimination, error, calibration, paired tests, subgroup and noise checks"),
                ("Envelope", "Evidence sufficiency, uncertainty, confidence, independent abstention"),
                ("Registry", "Lineage hash, model version, champion/candidate status"),
                ("Promotion gate", "Monitor/review/context only; treatment use is blocked"),
                ("Prediction trace", "Input evidence, missing modalities, output, version, timestamp"),
            ],
            columns=5,
        )
    )
    story.append(P("Model heads and safe boundaries", "Heading2Custom"))
    story.append(
        data_table(
            ["Head", "Engineering target", "Output", "Permitted interpretation"],
            [
                ["Response-pattern classification", "Synthetic binary response-pattern target", "Class probability and label", "Synthetic grouping for review; not personal benefit probability"],
                ["Response-score regression", "Synthetic continuous response score", "Point estimate and uncertainty band", "Simulator-target summary; not tumor response or prognosis"],
                ["Toxicity review signal", "Simulator-built support-review priority", "Review hint / abstention", "Shortcut-prone review signal; not toxicity diagnosis or grade"],
                ["Evidence sufficiency", "Availability and quality of required modalities", "Sufficient / insufficient plus missing data", "Whether the model should speak, not whether the patient is safe"],
            ],
            [42 * mm, 52 * mm, 34 * mm, 42 * mm],
        )
    )
    story.append(P("Predictor classes are not interchangeable", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "Imaging/report trends and clinician-reviewed imaging summaries carry the strongest synthetic response-monitoring context.",
                "CBC/labs, symptoms, treatment timing, interruptions, and interventions contribute monitoring and review context.",
                "ER/PR/HER2/Ki-67 are contextual biomarkers, not direct treatment-response proof in this prototype.",
                "Genetic records and family history are for genetic-counseling readiness and context, not mutation inference from scans.",
                "Tumor-marker trends are review context only and cannot prove recurrence or treatment failure.",
                "Treatment history is structured context, never a regimen recommendation target.",
            ]
        )
    )
    page_break(story)
    section(story, "09", "Classification and regression mathematics", "This section explains how the displayed synthetic model numbers are produced.")
    story.append(P("Classification", "Heading2Custom"))
    story.append(P("logit(p) = beta_0 + sum_j beta_j*x_j; p = 1 / (1 + exp(-logit(p)))", "Formula"))
    story.append(P("binary cross-entropy = -(1/N) * sum_i [y_i*log(p_i) + (1-y_i)*log(1-p_i)]", "Formula"))
    story.append(
        P(
            "Linear logistic regression provides a transparent baseline. Tree ensembles and neural candidates learn nonlinear interactions, but are promoted only when paired tests, calibration, robustness, and traceability support the change. A value such as 0.82 means the model assigned 82% probability to its synthetic class under the fitted data distribution. It does not mean an 82% personal chance of treatment success.",
        )
    )
    story.append(P("Regression", "Heading2Custom"))
    story.append(P("prediction = f(x); residual_i = y_i - prediction_i", "Formula"))
    story.append(P("MAE = (1/N) * sum_i |y_i - prediction_i|", "Formula"))
    story.append(P("R^2 = 1 - sum_i residual_i^2 / sum_i (y_i - mean(y))^2", "Formula"))
    story.append(
        P(
            "The response-score regressor estimates a synthetic target created by the simulator. MAE reports average absolute error in that target's units. R-squared reports explained synthetic variance. Neither metric establishes a clinically meaningful treatment-response score.",
        )
    )
    story.append(P("Hybrid display number", "Heading2Custom"))
    story.append(P("hybrid_signal = 0.65 * calibrated_class_probability + 0.35 * normalized_regression_score", "Formula"))
    story.append(
        callout(
            "Display meaning",
            "The hybrid signal is a UI synthesis of two synthetic model heads when both have sufficient evidence. It is not a validated health score. If one head abstains, the UI must disclose missing evidence instead of silently substituting certainty.",
            tone="pink",
        )
    )
    story.append(P("Monitoring context index", "Heading2Custom"))
    story.append(P("index = clamp(base_pattern - min(urgent_flags*12,35) - min(watch_flags*5,20) - min(symptom_severity*1.5,12) - synthetic_lab_penalty, 0, 100)", "Formula"))
    story.append(
        P(
            "The index is an engineering presentation score that begins with the synthetic response-pattern signal and applies capped deductions for review flags, symptom severity, and synthetic-lab provenance. It is explicitly not a personalized health score, clinical severity grade, prognosis, or triage rule.",
        )
    )
    page_break(story)
    section(story, "10", "Calibration, uncertainty, and abstention", "A model can rank well and still be overconfident; each property is measured separately.")
    heads = per_head.get("heads", {})
    classification = heads.get("response_classification", {})
    regression = heads.get("response_regression", {})
    toxicity = heads.get("toxicity", {})
    story.append(
        metric_row(
            [
                ("Synthetic classifier AUROC", _fmt(classification.get("auroc"), 4)),
                ("Synthetic classifier Brier", _fmt(classification.get("brier"), 4)),
                ("Synthetic classifier ECE", _fmt(classification.get("ece"), 4)),
                ("Synthetic regression MAE", _fmt(regression.get("mae"), 4)),
            ]
        )
    )
    story.append(P("Calibration metrics", "Heading2Custom"))
    story.append(P("Brier = (1/N) * sum_i (p_i - y_i)^2", "Formula"))
    story.append(P("ECE = sum_b (n_b/N) * |accuracy_b - confidence_b|", "Formula"))
    story.extend(
        bullets(
            [
                "AUROC measures ranking across thresholds, not probability accuracy and not clinical usefulness.",
                "Brier score measures squared probability error; lower is better, but the target is still synthetic.",
                "ECE summarizes bin-level confidence mismatch and depends on binning choices.",
                "Reliability intervals are included because a single calibration number can hide small-bin uncertainty.",
            ]
        )
    )
    story.append(P("Split-conformal interval", "Heading2Custom"))
    story.append(P("qhat = quantile_{ceil((n+1)*(1-alpha))/n}(|y_cal - prediction_cal|)", "Formula"))
    story.append(P("interval(x) = [prediction(x) - qhat, prediction(x) + qhat]", "Formula"))
    story.append(
        data_table(
            ["Conformal item", "Current synthetic artifact"],
            [
                ["Nominal coverage", _fmt(conformal.get("nominal_coverage"), 3)],
                ["Raw coverage", _fmt(conformal.get("raw_coverage"), 3)],
                ["Adjusted coverage", _fmt(conformal.get("adjusted_coverage"), 3)],
                ["Adjusted median band width", _fmt(conformal.get("adjusted_median_band_width"), 3)],
                ["Boundary", conformal.get("claim_boundary", "Synthetic-only interval calibration")],
            ],
            [55 * mm, 115 * mm],
        )
    )
    story.append(P("Abstention", "Heading2Custom"))
    story.append(P("predict only if evidence_sufficient AND confidence >= threshold AND not OOD; otherwise abstain", "Formula"))
    story.append(
        P(
            "Raising a confidence threshold usually reduces coverage while increasing retained-case accuracy. The project tracks that tradeoff explicitly; it does not hide difficult cases by reporting only the retained subset.",
        )
    )
    story.append(
        callout(
            "Toxicity warning",
            f"The toxicity head can show extremely high synthetic discrimination ({_fmt(toxicity.get('auroc'), 4)} in the current per-head artifact), but shortcut audits identify simulator-derived near-label-proxy risk. It remains review-hint-only and must not be presented as a clinical toxicity predictor.",
            tone="red",
        )
    )
    page_break(story)
