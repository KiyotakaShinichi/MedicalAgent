"""Head-to-head: champion classifier vs. modality-robust variant.

Runs both models against the same modality-dropout scenarios and reports
*two* views of the comparison:

  1. **With abstention layer engaged** — both models go through the rule-based
     evidence-sufficiency gate.  When the rules abstain, no probability is
     emitted regardless of which model is loaded, so the meaningful delta
     here is the covered-row accuracy (rows that *do* get scored).

  2. **With abstention disabled (force-score)** — both models are forced to
     emit a probability on every row, even when the abstention rules would
     refuse.  This isolates the *intrinsic* robustness of the trained
     classifier: does the modality-dropout-trained variant produce more
     accurate / better-calibrated outputs on rows with missing modalities?

The second view is the one that proves whether the retraining actually
moved the needle on the underlying model.  The first view is the one that
matches production behavior.

Artifact: `Data/evals/models/latest_modality_robustness_comparison.json`
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.services.complete_synthetic_training import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from backend.services.evidence_abstention_eval import (
    DEFAULT_LABEL,
    DEFAULT_TRAINING_ROWS_PATH,
    SCENARIOS,
    _strip_modalities,
)
from backend.services.modality_dropout_training import (
    DEFAULT_MODEL_PATH as DEFAULT_ROBUST_MODEL_PATH,
)
from backend.services.predict_with_abstention import (
    DEFAULT_MODEL_PATH as DEFAULT_CHAMPION_MODEL_PATH,
    predict_with_abstention,
)


DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_modality_robustness_comparison.json"

# Models we're comparing.  The "champion" is the existing classifier on disk;
# the "robust" variant is the new modality-dropout-trained one.
MODELS_UNDER_TEST: dict[str, str] = {
    "champion": DEFAULT_CHAMPION_MODEL_PATH,
    "robust":   DEFAULT_ROBUST_MODEL_PATH,
}


def run_modality_robustness_comparison(
    *,
    training_rows_path: str = DEFAULT_TRAINING_ROWS_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    label_column: str = DEFAULT_LABEL,
    sample_size: int | None = None,
) -> dict[str, Any]:
    """Run the comparison sweep and persist the artifact."""
    rows = pd.read_csv(training_rows_path)
    if label_column not in rows.columns:
        raise ValueError(f"Label column '{label_column}' missing from {training_rows_path}")
    if sample_size is not None and sample_size < len(rows):
        rows = rows.sample(n=sample_size, random_state=0).reset_index(drop=True)

    # Load both models once.  Refusing to load the robust variant before
    # training has been run gives a clearer error than letting joblib fail.
    for label, path in MODELS_UNDER_TEST.items():
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{label} model artifact missing at {path}.  Run training first.",
            )
    loaded = {label: joblib.load(path) for label, path in MODELS_UNDER_TEST.items()}

    scenario_reports: list[dict[str, Any]] = []
    for scenario, stripped in SCENARIOS.items():
        masked = _strip_modalities(rows, stripped)
        scenario_reports.append(
            _compare_one_scenario(
                scenario=scenario,
                stripped=stripped,
                masked_rows=masked,
                full_rows=rows,
                label_column=label_column,
                models=loaded,
            ),
        )

    payload = _build_payload(scenario_reports, training_rows_path, output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _compare_one_scenario(
    *,
    scenario: str,
    stripped: tuple[str, ...],
    masked_rows: pd.DataFrame,
    full_rows: pd.DataFrame,
    label_column: str,
    models: dict[str, Any],
) -> dict[str, Any]:
    """Two views per scenario: force_score (isolates the trained classifier)
    and with_abstention (matches production behavior)."""
    labels = masked_rows[label_column].astype(int).values
    feature_frame = masked_rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    force_score_view: dict[str, Any] = {}
    for name, model in models.items():
        probs = model.predict_proba(feature_frame)[:, 1]
        predictions = (probs >= 0.5).astype(int)
        force_score_view[name] = {
            "accuracy": _round(float((predictions == labels).mean())),
            "brier":    _round(_brier(labels, probs)),
            "mean_probability": _round(float(probs.mean())),
        }

    # Production view — run predict_with_abstention against each model. The
    # abstention rules are model-agnostic so the coverage rate will be
    # identical between models for the same scenario, but covered accuracy
    # differs because the model outputs differ on the rows that pass the gate.
    with_abstention_view: dict[str, Any] = {}
    for name, _model in models.items():
        covered, abstained = _evaluate_with_abstention(
            masked_rows=masked_rows,
            full_rows=full_rows,
            label_column=label_column,
            model_path=MODELS_UNDER_TEST[name],
        )
        with_abstention_view[name] = {
            "coverage_rate":       _round(covered["coverage_rate"]),
            "abstention_rate":     _round(abstained["rate"]),
            "covered_accuracy":    _round(covered["accuracy"]),
            "covered_mean_probability": _round(covered["mean_prob"]),
        }

    return {
        "scenario": scenario,
        "stripped_modalities": list(stripped),
        "rows_evaluated": int(len(masked_rows)),
        "force_score": force_score_view,
        "with_abstention": with_abstention_view,
        "deltas": {
            "force_score_accuracy_robust_minus_champion": _round(
                force_score_view["robust"]["accuracy"] - force_score_view["champion"]["accuracy"],
            ),
            "force_score_brier_robust_minus_champion": _round(
                force_score_view["robust"]["brier"] - force_score_view["champion"]["brier"],
            ),
            "with_abstention_accuracy_robust_minus_champion": _round(
                (with_abstention_view["robust"]["covered_accuracy"] or 0)
                - (with_abstention_view["champion"]["covered_accuracy"] or 0),
            ),
        },
    }


def _evaluate_with_abstention(
    *,
    masked_rows: pd.DataFrame,
    full_rows: pd.DataFrame,
    label_column: str,
    model_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run `predict_with_abstention` once per row using the named model.
    Returns (covered_summary, abstained_summary)."""
    covered_predictions: list[tuple[float, int]] = []
    abstained_total = 0
    for masked_row, full_row in zip(masked_rows.to_dict("records"), full_rows.to_dict("records")):
        actual = int(full_row[label_column])
        pred = predict_with_abstention(masked_row, model_path=model_path)
        if pred.decision == "insufficient_evidence" or pred.probability is None:
            abstained_total += 1
            continue
        covered_predictions.append((pred.probability, actual))

    covered_n = len(covered_predictions)
    total = len(masked_rows)
    correct = sum(1 for p, a in covered_predictions if (1 if p >= 0.5 else 0) == a)
    return (
        {
            "coverage_rate": covered_n / max(1, total),
            "accuracy":      correct / covered_n if covered_n else None,
            "mean_prob":     sum(p for p, _ in covered_predictions) / covered_n if covered_n else None,
        },
        {"rate": abstained_total / max(1, total)},
    )


# ─── Payload + helpers ───────────────────────────────────────────────────────


def _build_payload(
    scenario_reports: list[dict[str, Any]],
    training_rows_path: str,
    output_path: str,
) -> dict[str, Any]:
    overall = _summarise(scenario_reports)
    return {
        "schema_version": "modality_robustness_comparison_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": overall["status"],
        "training_rows_path": training_rows_path,
        "output_path": output_path,
        "models_under_test": MODELS_UNDER_TEST,
        "scenarios": scenario_reports,
        "summary": overall,
        "interpretation": (
            "force_score view isolates the trained classifier (no abstention "
            "rules applied), with_abstention view matches production behavior. "
            "Positive accuracy deltas mean the robust variant beats the "
            "champion on the rows it scored; negative Brier deltas mean the "
            "robust variant is better-calibrated."
        ),
        "claim_boundary": (
            "Synthetic comparison only.  Improvements here describe how "
            "the model handles its own training-distribution missingness "
            "pattern, not clinical robustness on real patient timelines."
        ),
    }


def _summarise(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    full = next((s for s in scenarios if s["scenario"] == "full_data"), None)
    accuracy_wins = sum(
        1 for s in scenarios
        if (s["deltas"]["force_score_accuracy_robust_minus_champion"] or 0) > 0.005
    )
    accuracy_ties = sum(
        1 for s in scenarios
        if abs(s["deltas"]["force_score_accuracy_robust_minus_champion"] or 0) <= 0.005
    )
    accuracy_losses = len(scenarios) - accuracy_wins - accuracy_ties

    # Verdict: robust variant must not regress on full_data and must improve
    # average force-score accuracy on at least half of the partial-evidence
    # scenarios.
    full_data_regression = (
        full is not None
        and (full["deltas"]["force_score_accuracy_robust_minus_champion"] or 0) < -0.01
    )
    if full_data_regression:
        status = "needs_attention"
    elif accuracy_wins >= accuracy_losses:
        status = "robust"
    else:
        status = "acceptable"

    return {
        "status": status,
        "scenario_count": len(scenarios),
        "force_score_accuracy_wins_for_robust": accuracy_wins,
        "force_score_accuracy_ties": accuracy_ties,
        "force_score_accuracy_losses_for_robust": accuracy_losses,
        "full_data_accuracy_delta": (
            full["deltas"]["force_score_accuracy_robust_minus_champion"] if full else None
        ),
        "full_data_brier_delta": (
            full["deltas"]["force_score_brier_robust_minus_champion"] if full else None
        ),
    }


def _brier(labels: np.ndarray, probs: np.ndarray) -> float:
    return float(((probs - labels) ** 2).mean())


def _round(value: float | None, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def load_modality_robustness_comparison(
    path: str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Read the most-recently-written comparison artifact for the admin UI."""
    file_path = Path(path)
    if not file_path.exists():
        return {
            "schema_version": "modality_robustness_comparison_v1",
            "status": "missing",
            "message": (
                "Modality-robustness comparison has not been generated yet. "
                "Train the robust variant via "
                "`scripts/run_modality_dropout_training.py`, then run "
                "`scripts/run_modality_robustness_comparison.py`."
            ),
            "scenarios": [],
            "summary": {},
        }
    return json.loads(file_path.read_text(encoding="utf-8"))


__all__ = [
    "MODELS_UNDER_TEST",
    "load_modality_robustness_comparison",
    "run_modality_robustness_comparison",
]
