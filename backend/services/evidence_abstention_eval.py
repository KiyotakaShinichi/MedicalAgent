"""Evidence-abstention evaluation harness.

What this measures
------------------
Given the trained champion model + the evidence-sufficiency layer, this
service simulates patient timelines with various missing-modality patterns
and measures:

  - abstention rate per scenario (what fraction of rows the system refuses to
    score),
  - false-abstention rate (rows we abstain on where the underlying model
    *would* have been correct — too cautious),
  - covered-row accuracy (accuracy on the rows we choose to score),
  - covered-row mean probability (how confident the model is when it answers),
  - selective-risk curve at the row level.

The point is not to optimise any one number.  A senior reviewer reads this
artifact and asks: "Do you over-abstain on easy cases? Do you under-abstain
on cases where you should have refused?  How does coverage trade off against
correctness?"  The eval shows both sides.

Scenarios
---------
The scenarios are deliberately named after the clinical situation they
represent so the dashboard can show the trade-off in human terms.  Each
scenario specifies a list of modality groups to *strip* before running the
prediction.

  - full_data
  - no_imaging                       — patient has CBC + symptoms but no MRI/CT
  - no_nadir_cbc                     — pre + recovery only, mid-cycle data missing
  - no_recovery_cbc                  — cycle still in progress
  - imaging_only                     — every CBC/symptom column stripped
  - cbc_pre_only                     — first-visit profile, no longitudinal data
  - demographics_only                — record skeleton, no real evidence
  - symptoms_only                    — patient-reported only

Artifact path: ``Data/evals/models/latest_evidence_abstention_eval.json``
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from backend.services.complete_synthetic_training import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from backend.services.evidence_sufficiency import MODALITY_GROUPS
from backend.services.predict_with_abstention import (
    DEFAULT_CALIBRATOR_PATH,
    DEFAULT_MODEL_PATH,
    predict_with_abstention,
)


DEFAULT_TRAINING_ROWS_PATH = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_evidence_abstention_eval.json"
DEFAULT_LABEL = "treatment_success_binary"


# Scenario name → modality groups stripped from the row before prediction.
# `full_data` is the no-op baseline so coverage/accuracy can be compared
# against the unfiltered case.
SCENARIOS: dict[str, tuple[str, ...]] = {
    "full_data":          tuple(),
    "no_imaging":         ("imaging",),
    "no_nadir_cbc":       ("cbc_nadir",),
    "no_recovery_cbc":    ("cbc_recovery",),
    "imaging_only":       ("cbc_pre", "cbc_nadir", "cbc_recovery", "symptoms", "interventions"),
    "cbc_pre_only":       ("cbc_nadir", "cbc_recovery", "imaging", "symptoms", "interventions"),
    "demographics_only":  ("cbc_pre", "cbc_nadir", "cbc_recovery", "imaging", "symptoms", "interventions"),
    "symptoms_only":      ("cbc_pre", "cbc_nadir", "cbc_recovery", "imaging", "interventions"),
}


def run_evidence_abstention_eval(
    training_rows_path: str = DEFAULT_TRAINING_ROWS_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    calibrator_path: str | None = DEFAULT_CALIBRATOR_PATH,
    label_column: str = DEFAULT_LABEL,
    sample_size: int | None = None,
) -> dict[str, Any]:
    """Run the abstention sweep, write the artifact, and return the payload."""
    rows = pd.read_csv(training_rows_path)
    if label_column not in rows.columns:
        raise ValueError(f"Label column '{label_column}' not present in {training_rows_path}")
    if sample_size is not None and sample_size < len(rows):
        rows = rows.sample(n=sample_size, random_state=0).reset_index(drop=True)

    scenario_reports: list[dict[str, Any]] = []
    for scenario, stripped in SCENARIOS.items():
        scenario_reports.append(
            _evaluate_scenario(
                scenario=scenario,
                rows=rows,
                stripped_groups=stripped,
                label_column=label_column,
                model_path=model_path,
                calibrator_path=calibrator_path,
            ),
        )

    payload = _build_payload(
        scenario_reports=scenario_reports,
        training_rows_path=training_rows_path,
        model_path=model_path,
        calibrator_path=calibrator_path,
        label_column=label_column,
        evaluated_rows=len(rows),
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _evaluate_scenario(
    *,
    scenario: str,
    rows: pd.DataFrame,
    stripped_groups: tuple[str, ...],
    label_column: str,
    model_path: str,
    calibrator_path: str | None,
) -> dict[str, Any]:
    """Run one scenario.  Counts rows by decision, computes covered-row
    accuracy, and reports a tiny calibration histogram."""
    masked = _strip_modalities(rows, stripped_groups)
    decisions = Counter()
    covered_predictions: list[tuple[float, int]] = []
    abstained_with_correct_underlying_decision = 0
    abstained_total = 0

    # We compute "would the underlying classifier have been right?" by also
    # running prediction on the *unmasked* row.  This lets us report false
    # abstention — cases where we refused to answer but the full-data model
    # was actually correct.
    for _, masked_row, full_row in _iter_pairs(rows, masked):
        actual = int(full_row[label_column])
        prediction = predict_with_abstention(
            masked_row,
            model_path=model_path,
            calibrator_path=calibrator_path,
        )
        decisions[prediction.decision] += 1
        if prediction.decision == "insufficient_evidence":
            abstained_total += 1
            # Run the model on the unmasked row to see what it WOULD have said.
            full_pred = predict_with_abstention(
                full_row,
                model_path=model_path,
                calibrator_path=calibrator_path,
            )
            if full_pred.probability is not None:
                predicted_class = 1 if full_pred.probability >= 0.5 else 0
                if predicted_class == actual:
                    abstained_with_correct_underlying_decision += 1
        elif prediction.probability is not None:
            covered_predictions.append((prediction.probability, actual))

    covered_n = len(covered_predictions)
    correct_when_covered = sum(
        1 for p, a in covered_predictions if (1 if p >= 0.5 else 0) == a
    )
    coverage_rate = covered_n / max(1, len(masked))
    abstention_rate = abstained_total / max(1, len(masked))
    false_abstention_rate = (
        abstained_with_correct_underlying_decision / max(1, abstained_total)
        if abstained_total > 0 else None
    )

    return {
        "scenario": scenario,
        "stripped_modalities": list(stripped_groups),
        "rows_evaluated": int(len(masked)),
        "coverage_rate": _round(coverage_rate),
        "abstention_rate": _round(abstention_rate),
        "false_abstention_rate": _round(false_abstention_rate),
        "covered_accuracy": _round(correct_when_covered / covered_n) if covered_n > 0 else None,
        "decision_counts": dict(decisions),
        "covered_mean_probability": _round(
            sum(p for p, _ in covered_predictions) / covered_n
        ) if covered_n > 0 else None,
        "calibration_bins": _calibration_bins(covered_predictions),
    }


def _iter_pairs(rows: pd.DataFrame, masked: pd.DataFrame):
    """Yield ``(index, masked_row, full_row)`` triples in lockstep."""
    full_records = rows.to_dict("records")
    masked_records = masked.to_dict("records")
    for idx, (m, f) in enumerate(zip(masked_records, full_records)):
        yield idx, m, f


def _strip_modalities(rows: pd.DataFrame, groups: tuple[str, ...]) -> pd.DataFrame:
    """Return a copy of `rows` with every column belonging to the given
    modality groups set to NaN (numerics) or "" (categoricals)."""
    if not groups:
        return rows.copy()
    masked = rows.copy()
    for group in groups:
        for column in MODALITY_GROUPS.get(group, ()):
            if column not in masked.columns:
                continue
            if column in NUMERIC_FEATURES:
                masked[column] = float("nan")
            elif column in CATEGORICAL_FEATURES:
                masked[column] = ""
            else:
                masked[column] = float("nan")
    return masked


def _calibration_bins(predictions: list[tuple[float, int]], n_bins: int = 5) -> list[dict[str, Any]]:
    """Reliability histogram: per equal-width probability bin, report the
    mean predicted probability and the observed positive rate.  Used by the
    dashboard to flag bands where the model is over- or under-confident."""
    if not predictions:
        return []
    bins: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        in_bin = [(p, a) for p, a in predictions if (lo <= p < hi if i < n_bins - 1 else lo <= p <= hi)]
        if not in_bin:
            bins.append({"range": f"{lo:.2f}-{hi:.2f}", "count": 0, "mean_predicted": None, "observed_rate": None})
            continue
        bins.append({
            "range": f"{lo:.2f}-{hi:.2f}",
            "count": len(in_bin),
            "mean_predicted": _round(sum(p for p, _ in in_bin) / len(in_bin)),
            "observed_rate": _round(sum(a for _, a in in_bin) / len(in_bin)),
        })
    return bins


def _build_payload(
    *,
    scenario_reports: list[dict[str, Any]],
    training_rows_path: str,
    model_path: str,
    calibrator_path: str | None,
    label_column: str,
    evaluated_rows: int,
) -> dict[str, Any]:
    summary = _summarise(scenario_reports)
    return {
        "schema_version": "evidence_abstention_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": summary["overall_status"],
        "training_rows_path": training_rows_path,
        "model_path": model_path,
        "calibrator_path": calibrator_path,
        "label_column": label_column,
        "rows_evaluated": evaluated_rows,
        "summary": summary,
        "scenarios": scenario_reports,
        "interpretation": (
            "Coverage is the fraction of rows the system chose to score; "
            "abstention rate is the complement.  False abstention rate is "
            "the fraction of abstained rows where the underlying model "
            "would have been correct — high values indicate the abstention "
            "rules are too aggressive."
        ),
        "claim_boundary": (
            "Synthetic evaluation only.  Coverage and accuracy numbers "
            "cannot be interpreted as clinical performance — they describe "
            "the engineering trade-off between confidence and refusal."
        ),
    }


def _summarise(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Cross-scenario summary used by the benchmark registry.  Marks the
    overall status `acceptable` when full_data has high coverage AND the
    demographics_only scenario fully abstains."""
    by_scenario = {r["scenario"]: r for r in reports}
    full = by_scenario.get("full_data", {})
    demo_only = by_scenario.get("demographics_only", {})
    full_coverage = full.get("coverage_rate") or 0.0
    demo_abstention = demo_only.get("abstention_rate") or 0.0

    if full_coverage >= 0.95 and demo_abstention >= 0.95:
        status = "strong"
    elif full_coverage >= 0.80 and demo_abstention >= 0.80:
        status = "acceptable"
    else:
        status = "needs_attention"

    abstention_rates = {r["scenario"]: r["abstention_rate"] for r in reports}
    coverage_rates = {r["scenario"]: r["coverage_rate"] for r in reports}
    covered_accuracies = {
        r["scenario"]: r["covered_accuracy"] for r in reports
    }
    return {
        "overall_status": status,
        "full_data_coverage_rate": full.get("coverage_rate"),
        "full_data_covered_accuracy": full.get("covered_accuracy"),
        "demographics_only_abstention_rate": demo_only.get("abstention_rate"),
        "abstention_rates_by_scenario": abstention_rates,
        "coverage_rates_by_scenario": coverage_rates,
        "covered_accuracy_by_scenario": covered_accuracies,
        "scenario_count": len(reports),
    }


def _round(value: float | None, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def load_evidence_abstention_eval(output_path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    """Read the cached artifact for the admin GET endpoint."""
    path = Path(output_path)
    if not path.exists():
        return {
            "schema_version": "evidence_abstention_eval_v1",
            "status": "missing",
            "message": (
                "Abstention eval has not been generated yet. Run "
                "`scripts/run_evidence_abstention_eval.py` or POST to "
                "/admin/evidence-abstention-eval to produce it."
            ),
            "scenarios": [],
            "summary": {},
        }
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "SCENARIOS",
    "run_evidence_abstention_eval",
    "load_evidence_abstention_eval",
]
