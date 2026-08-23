"""Sections 11-12: statistical testing and temporal design, then
missingness, robustness, and out-of-distribution behaviour.

Extracted verbatim from ``build_story`` in
``scripts/generate_project_guide_pdf.py``, which had grown to 1032 lines in a
single function. Flowable content and ordering are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from reportlab.lib.units import mm
from scripts.project_guide.components import P, bullets, callout, data_table, flow_diagram, page_break, section
from scripts.project_guide.evidence import _fmt

if TYPE_CHECKING:
    from scripts.project_guide.evidence import Evidence

def build(story: list[Any], ev: Evidence) -> None:
    """Append this module's sections to `story`, in order."""
    temporal = ev.temporal
    temporal_cv = ev.temporal_cv
    paired = ev.paired

    section(story, "11", "Statistical testing and temporal design", "The statistics test engineering comparisons inside the synthetic experiment; they do not repair target validity.")
    story.append(P("Patient-level temporal cross-validation", "Heading2Custom"))
    story.append(
        flow_diagram(
            [
                ("Group", "All cycles from one synthetic patient remain together"),
                ("Order", "Train precedes validation in time"),
                ("Walk forward", "Repeat across sequential patient-time folds"),
                ("Audit", "Check patient overlap and temporal violations"),
            ]
        )
    )
    story.append(
        data_table(
            ["Temporal-CV field", "Current artifact", "Interpretation"],
            [
                ["Configured folds", _fmt(temporal.get("n_folds")), f"{_fmt(temporal_cv.get('n_folds_with_auc'))} folds reported AUROC"],
                ["Mean AUROC", _fmt(temporal_cv.get("roc_auc_mean"), 6), "Saturated synthetic result; not clinical discrimination"],
                ["Brier", _fmt(temporal_cv.get("brier_mean"), 6), "Synthetic probability error"],
                ["Patient overlap", _fmt(temporal_cv.get("patient_overlap_pairs")), "Expected zero"],
                ["Temporal violations", _fmt(temporal_cv.get("temporal_violations")), "Must remain visible and investigated"],
            ],
            [40 * mm, 35 * mm, 95 * mm],
        )
    )
    story.append(P("Exact paired classification comparison", "Heading2Custom"))
    story.append(P("McNemar uses discordant pairs b and c: chi_square = (|b-c|-1)^2 / (b+c)", "Formula"))
    story.append(
        P(
            "Because candidate and champion predict the same held-out rows, McNemar's test asks whether one model wins on significantly more discordant examples. It is stronger than comparing two rounded accuracy values but still inherits the synthetic dataset's limitations.",
        )
    )
    class_rows: list[list[Any]] = []
    for item in paired.get("classification_comparisons", paired.get("classification", []))[:4]:
        class_rows.append(
            [
                item.get("candidate", item.get("candidate_model", "candidate")),
                _fmt(item.get("champion_accuracy"), 4),
                _fmt(item.get("candidate_accuracy"), 4),
                _fmt(item.get("accuracy_delta_champion_minus_candidate"), 4),
                _fmt(item.get("p_value"), 6),
            ]
        )
    if class_rows:
        story.append(data_table(["Candidate", "Champion acc.", "Candidate acc.", "Delta", "Exact p"], class_rows, [62 * mm, 27 * mm, 27 * mm, 24 * mm, 30 * mm]))
    story.append(P("Paired bootstrap for regression", "Heading2Custom"))
    story.append(P("For each resample: delta_MAE* = MAE(champion*) - MAE(candidate*)", "Formula"))
    story.append(
        P(
            "Rows are resampled as matched pairs, preserving that both models saw the same examples. The empirical percentile interval estimates uncertainty in the MAE difference. If the interval crosses zero, the comparison does not clearly establish a winner.",
        )
    )
    reg_rows: list[list[Any]] = []
    for item in paired.get("regression_comparisons", paired.get("regression", []))[:4]:
        ci = item.get("bootstrap_ci", item.get("mae_delta_ci", []))
        if isinstance(ci, dict):
            ci_text = f"[{_fmt(ci.get('lower'), 3)}, {_fmt(ci.get('upper'), 3)}]"
        elif isinstance(ci, list) and len(ci) >= 2:
            ci_text = f"[{_fmt(ci[0], 3)}, {_fmt(ci[1], 3)}]"
        elif item.get("ci_low") is not None and item.get("ci_high") is not None:
            ci_text = f"[{_fmt(item.get('ci_low'), 3)}, {_fmt(item.get('ci_high'), 3)}]"
        else:
            ci_text = "not reported"
        reg_rows.append(
            [
                item.get("candidate", item.get("candidate_model", "candidate")),
                _fmt(item.get("champion_mae"), 3),
                _fmt(item.get("candidate_mae"), 3),
                _fmt(item.get("mae_delta_champion_minus_candidate"), 3),
                ci_text,
            ]
        )
    if reg_rows:
        story.append(data_table(["Candidate", "Champion MAE", "Candidate MAE", "Delta", "Bootstrap CI"], reg_rows, [62 * mm, 27 * mm, 27 * mm, 24 * mm, 30 * mm]))
    page_break(story)
    section(story, "12", "Missingness, robustness, and OOD", "Missing data is modeled, disclosed, and allowed to force abstention; it is not replaced by false certainty.")
    story.append(P("Missingness representation", "Heading2Custom"))
    story.append(P("x'_j = impute(x_j); m_j = 1[x_j is missing]; model_input = [x', m]", "Formula"))
    story.extend(
        bullets(
            [
                "Imputed values and missingness indicators are separate features so the model can learn synthetic missingness patterns.",
                "Modality-dropout training simulates absent imaging, CBC, symptoms, demographics, or intervention context.",
                "Each model head has minimum-evidence rules; a classification head can speak while regression abstains, or vice versa.",
                "The response includes modalities present, modalities missing, sufficiency, uncertainty, and an abstention reason.",
                "This demonstrates graceful degradation in the simulator; it does not prove safe missing-data handling for real patients.",
            ]
        )
    )
    story.append(P("OOD and quality gates", "Heading2Custom"))
    story.append(
        data_table(
            ["Gate", "Question", "Safe behavior"],
            [
                ["Schema", "Are fields, units, dates, and categories valid?", "Reject or request correction"],
                ["Range", "Is a value outside configured engineering expectations?", "Flag quality issue; do not infer a diagnosis"],
                ["Distribution", "Is the feature vector far from synthetic training support?", "Mark OOD and abstain/review"],
                ["Modality", "Are required evidence classes absent?", "Independent head abstention"],
                ["Shortcut", "Does a feature behave like a target proxy?", "Block promotion and document the weakness"],
                ["Subgroup floor", "Does performance collapse in a synthetic subgroup?", "Hold promotion and report the slice"],
            ],
            [32 * mm, 70 * mm, 68 * mm],
        )
    )
    story.append(
        callout(
            "Critical distinction",
            "A missingness-aware model can return a number when inputs are incomplete, but it should do so only when that head's evidence rules allow it. Confidence is not a license to invent absent clinical context. The safest valid output may be 'insufficient evidence' with a list of missing modalities.",
            tone="amber",
        )
    )
    page_break(story)
