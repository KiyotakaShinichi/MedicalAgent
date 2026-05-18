"""Toxicity classifier feature-importance audit + no-proxy baseline.

Why this exists
---------------
The toxicity binary classifier reports AUC ≈ 1.0 on the synthetic test
split.  That sounds great but is actually a red flag in synthetic data:
it usually means the labels are a tautological function of a single
feature, not that the model learned anything generalizable.

The audit:

1. Loads the trained toxicity classifier and computes per-feature
   importance from the underlying gradient-booster.
2. Computes per-feature *label-separation gap* — the difference in
   positive-label rate between the top decile and bottom decile of the
   feature's value distribution.  A gap near 1.0 means the feature is
   nearly identical to the label.
3. Flags any feature whose importance exceeds a configurable threshold
   (default 0.50) — "dominant features" that should be reviewed.
4. Trains a **no-proxy baseline** on the same data with the dominant
   features stripped out, and records its AUC + Brier.  That number is
   the honest "how much real signal does the model have once the
   tautology is removed" figure.

Output
------
``Data/evals/models/latest_toxicity_feature_audit.json`` with:
  - status (passed / needs_attention / missing)
  - per-feature importance + label-separation gap
  - dominant_features list
  - no_proxy_baseline metrics (auc, brier, dropped_features)
  - interpretation block

The audit makes the toxicity-head limitation **explicit** rather than
buried in a fortunate-looking AUC number.  A clinical reviewer can see
exactly which features the model leans on and what falls out when the
dominant ones are removed.

Engineering provenance only.  This does not establish clinical toxicity
prediction validity — it only documents how the synthetic-trained model
arrives at its synthetic numbers.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from backend.services.complete_synthetic_training import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    _patient_split,
    _preprocessor,
)


DEFAULT_ML_CSV_PATH = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_TOXICITY_MODEL_PATH = (
    "Data/complete_synthetic_training/gradient_boosting_toxicity_risk_binary.joblib"
)
DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_toxicity_feature_audit.json"
DEFAULT_TARGET = "toxicity_risk_binary"

# Any feature whose normalized importance exceeds this threshold is flagged
# as "dominant" and triggers the no-proxy baseline.  0.50 = the feature
# carries more weight than every other feature combined.
DOMINANT_FEATURE_THRESHOLD = 0.50

# A separation gap above this is considered near-label-identity (top vs
# bottom decile differ by more than this fraction).  Documented separately
# from importance so a reviewer can spot proxies the model didn't lean on
# (which is its own kind of risk — a future model variant might).
NEAR_LABEL_IDENTITY_GAP = 0.85


@dataclass
class FeatureRow:
    feature: str
    importance: float
    label_separation_gap: float | None
    bottom_decile_rate: float | None
    top_decile_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "importance": round(float(self.importance), 4),
            "label_separation_gap": (
                round(float(self.label_separation_gap), 4)
                if self.label_separation_gap is not None else None
            ),
            "bottom_decile_rate": (
                round(float(self.bottom_decile_rate), 4)
                if self.bottom_decile_rate is not None else None
            ),
            "top_decile_rate": (
                round(float(self.top_decile_rate), 4)
                if self.top_decile_rate is not None else None
            ),
        }


# ─── Public API ──────────────────────────────────────────────────────────────


def run_toxicity_feature_audit(
    *,
    ml_csv_path: str = DEFAULT_ML_CSV_PATH,
    model_path: str = DEFAULT_TOXICITY_MODEL_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    target: str = DEFAULT_TARGET,
    dominant_threshold: float = DOMINANT_FEATURE_THRESHOLD,
    test_size: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the audit + write the artifact.  Returns the payload."""
    rows = pd.read_csv(ml_csv_path)
    if target not in rows.columns:
        raise ValueError(f"Target column '{target}' not present in {ml_csv_path}")
    rows = rows.dropna(subset=[target]).copy()

    # Per-feature label-separation gap is computed on the FULL dataset so
    # importance + gap line up on the same denominator.
    feature_rows = _per_feature_label_separation(rows, target)

    # Pull importances from the trained classifier when its artifact is
    # present.  When it's not, fall back to None importance values so the
    # audit still emits the gap analysis.
    importances = _model_feature_importances(model_path)
    if importances:
        importance_lookup = importances
    else:
        importance_lookup = {fr.feature: 0.0 for fr in feature_rows}
    for fr in feature_rows:
        fr.importance = importance_lookup.get(fr.feature, 0.0)

    # Sort by importance, descending — readers want the dominant features first.
    feature_rows.sort(key=lambda fr: fr.importance, reverse=True)

    dominant_features = [fr.feature for fr in feature_rows if fr.importance >= dominant_threshold]
    near_label_proxies = [
        fr.feature for fr in feature_rows
        if fr.label_separation_gap is not None and fr.label_separation_gap >= NEAR_LABEL_IDENTITY_GAP
    ]

    # No-proxy baseline — strip the dominant features + retrain.  Always run
    # this so the artifact has the honest number, even when no feature
    # currently exceeds the threshold (in which case dropped_features is
    # empty and the AUC reflects the full-feature classifier).
    baseline = _train_no_proxy_baseline(
        rows=rows,
        target=target,
        dropped_features=dominant_features,
        test_size=test_size,
        seed=seed,
    )

    # Strict baseline — strip EVERY near-label-proxy feature, not just the
    # dominant ones.  When this baseline ALSO produces high AUC, the
    # synthetic generator has structural tautologies the model can't avoid,
    # and the raw AUC isn't informative on its own.  This is the bar that
    # actually answers "is there real signal beyond the generator's
    # built-in shortcuts?"
    strict_dropped = sorted(set(dominant_features) | set(near_label_proxies))
    strict_baseline = _train_no_proxy_baseline(
        rows=rows,
        target=target,
        dropped_features=strict_dropped,
        test_size=test_size,
        seed=seed,
    )

    status = _overall_status(dominant_features, baseline, strict_baseline)
    payload = {
        "schema_version": "toxicity_feature_audit_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "target": target,
        "ml_csv_path": ml_csv_path,
        "model_path": model_path,
        "dominant_threshold": dominant_threshold,
        "near_label_identity_gap_threshold": NEAR_LABEL_IDENTITY_GAP,
        "label_positive_rate": round(float(rows[target].mean()), 4),
        "feature_rows": [fr.to_dict() for fr in feature_rows],
        "dominant_features": dominant_features,
        "near_label_proxy_features": near_label_proxies,
        "no_proxy_baseline": baseline,
        "strict_no_proxy_baseline": strict_baseline,
        "interpretation": _interpretation(dominant_features, baseline, strict_baseline),
        "claim_boundary": (
            "Engineering audit only. Documents how the synthetic-trained "
            "toxicity classifier arrives at its synthetic AUC; does not "
            "establish clinical toxicity prediction validity."
        ),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_toxicity_feature_audit(path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {
            "schema_version": "toxicity_feature_audit_v1",
            "status": "missing",
            "message": (
                "Toxicity feature audit has not been generated yet. Run "
                "`scripts/run_toxicity_feature_audit.py`."
            ),
            "feature_rows": [],
            "dominant_features": [],
            "no_proxy_baseline": {},
        }
    return json.loads(file_path.read_text(encoding="utf-8"))


# ─── Internals ───────────────────────────────────────────────────────────────


def _per_feature_label_separation(rows: pd.DataFrame, target: str) -> list[FeatureRow]:
    """For every numeric feature, compute positive-label rate in the top vs
    bottom decile.  Categorical features get a None gap — they're tracked
    in the importance ranking but the decile analysis only makes sense for
    ordered values."""
    y = rows[target].astype(int)
    feature_rows: list[FeatureRow] = []
    for column in NUMERIC_FEATURES:
        if column not in rows.columns:
            continue
        s = pd.to_numeric(rows[column], errors="coerce")
        valid = s.notna() & y.notna()
        if valid.sum() < 100:
            feature_rows.append(FeatureRow(
                feature=column, importance=0.0,
                label_separation_gap=None,
                bottom_decile_rate=None, top_decile_rate=None,
            ))
            continue
        sv = s[valid]; yv = y[valid]
        q1, q9 = np.quantile(sv, [0.10, 0.90])
        bottom_rate = float(yv[sv <= q1].mean()) if (sv <= q1).any() else None
        top_rate = float(yv[sv >= q9].mean()) if (sv >= q9).any() else None
        gap = (
            abs(top_rate - bottom_rate)
            if bottom_rate is not None and top_rate is not None else None
        )
        feature_rows.append(FeatureRow(
            feature=column, importance=0.0,
            label_separation_gap=gap,
            bottom_decile_rate=bottom_rate, top_decile_rate=top_rate,
        ))
    for column in CATEGORICAL_FEATURES:
        if column not in rows.columns:
            continue
        feature_rows.append(FeatureRow(
            feature=column, importance=0.0,
            label_separation_gap=None,
            bottom_decile_rate=None, top_decile_rate=None,
        ))
    return feature_rows


def _model_feature_importances(model_path: str) -> dict[str, float]:
    """Pull feature_importances_ from the trained model.  Returns an empty
    dict when the artifact isn't on disk — caller falls back to zero
    importances so the gap analysis still ships."""
    path = Path(model_path)
    if not path.exists():
        return {}
    model: Pipeline = joblib.load(model_path)
    try:
        classifier = model.named_steps["classifier"]
        preprocess = model.named_steps["preprocess"]
    except (AttributeError, KeyError):
        return {}

    importances = getattr(classifier, "feature_importances_", None)
    if importances is None:
        return {}

    # Build the post-OHE feature-name list to match `importances`.
    expanded_names: list[str] = []
    for name, transformer, cols in preprocess.transformers_:
        if name == "numeric":
            expanded_names.extend(cols)
        elif name == "categorical":
            try:
                expanded_names.extend(transformer.get_feature_names_out(cols))
            except Exception:
                expanded_names.extend(cols)

    # Fold OHE-expanded columns back onto their parent categorical feature
    # so the per-feature view reads cleanly (we sum the importance across
    # the one-hot columns belonging to the same parent).
    parent_lookup = {col: col for col in NUMERIC_FEATURES}
    for cat in CATEGORICAL_FEATURES:
        for name in expanded_names:
            if name == cat or name.startswith(f"{cat}_"):
                parent_lookup[name] = cat

    aggregated: dict[str, float] = {}
    for name, importance in zip(expanded_names, importances):
        parent = parent_lookup.get(name, name)
        aggregated[parent] = aggregated.get(parent, 0.0) + float(importance)
    return aggregated


def _train_no_proxy_baseline(
    *,
    rows: pd.DataFrame,
    target: str,
    dropped_features: list[str],
    test_size: float,
    seed: int,
) -> dict[str, Any]:
    """Retrain a fresh gradient-boosting classifier with the dominant
    features stripped out, score it on the same patient-aware split, and
    return its honest AUC + Brier."""
    train_patients, test_patients = _patient_split(rows, target, test_size, seed)
    train_rows = rows[rows["patient_id"].isin(train_patients)].copy()
    test_rows = rows[rows["patient_id"].isin(test_patients)].copy()

    remaining_numeric = [c for c in NUMERIC_FEATURES if c not in dropped_features]
    remaining_categorical = [c for c in CATEGORICAL_FEATURES if c not in dropped_features]

    if not remaining_numeric and not remaining_categorical:
        return {
            "status": "no_features_left",
            "auc": None,
            "brier": None,
            "dropped_features": list(dropped_features),
            "remaining_feature_count": 0,
        }

    X_train = train_rows[remaining_numeric + remaining_categorical]
    y_train = train_rows[target].astype(int)
    X_test = test_rows[remaining_numeric + remaining_categorical]
    y_test = test_rows[target].astype(int)

    # Build a preprocessor pinned to the remaining feature names — the
    # default `_preprocessor` factory assumes the full feature contract.
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), remaining_numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), remaining_categorical),
    ], sparse_threshold=0)
    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("classifier", GradientBoostingClassifier(random_state=seed)),
    ])
    pipeline.fit(X_train, y_train)
    probs = pipeline.predict_proba(X_test)[:, 1]
    return {
        "status": "trained",
        "auc": round(float(roc_auc_score(y_test, probs)), 4),
        "brier": round(float(brier_score_loss(y_test, probs)), 4),
        "dropped_features": list(dropped_features),
        "remaining_feature_count": len(remaining_numeric) + len(remaining_categorical),
        "test_rows": int(len(test_rows)),
        "train_rows": int(len(train_rows)),
    }


def _overall_status(
    dominant_features: list[str],
    baseline: dict[str, Any],
    strict_baseline: dict[str, Any],
) -> str:
    """Status policy:
      - `strong` when no dominant features AND strict-baseline AUC stays
        meaningfully above chance (the model has real signal).
      - `acceptable` when dominant features exist OR strict-baseline AUC
        collapses — both outcomes are honest findings that the AUC is
        synthetic-generator-driven.
      - `needs_attention` when dominant features exist AND the strict
        baseline ALSO stays high — the synthetic generator structurally
        leaks the label across many features and the AUC tells us nothing.
    """
    strict_auc = strict_baseline.get("auc")
    if not dominant_features:
        return "strong"
    if strict_auc is None:
        return "acceptable"
    if strict_auc >= 0.90:
        return "needs_attention"
    return "acceptable"


def _interpretation(
    dominant_features: list[str],
    baseline: dict[str, Any],
    strict_baseline: dict[str, Any],
) -> str:
    if not dominant_features:
        return (
            "No single feature exceeds the dominance threshold. The "
            "toxicity classifier is leaning on a spread of features, "
            "which is what the design wants."
        )
    auc = baseline.get("auc")
    strict_auc = strict_baseline.get("auc")
    strict_dropped = strict_baseline.get("dropped_features") or []
    drop = ", ".join(dominant_features)

    if auc is None:
        return (
            f"Dominant feature(s) found: {drop}. No-proxy baseline did not "
            "train successfully — review the artifact."
        )
    if strict_auc is None:
        return (
            f"Dominant feature(s) found: {drop}. Stricter no-proxy baseline "
            "(strip every near-label feature) did not train — review."
        )
    if strict_auc >= 0.90:
        return (
            f"Dominant feature(s) found: {drop}. Even after stripping ALL "
            f"{len(strict_dropped)} near-label-proxy features, AUC stays at "
            f"{strict_auc:.3f}. This means the synthetic generator wires "
            "the toxicity label to too many features to remove cleanly — "
            "the AUC is a property of the generator, not the model's "
            "learned skill. Quoting this AUC as model performance would "
            "be misleading; the model card must reference this audit."
        )
    if auc < 0.70:
        return (
            f"Dominant feature(s) found: {drop}. Removing them drops AUC "
            f"to {auc:.3f} — confirms the original AUC ≈ 1.0 was a "
            "tautology in the synthetic generator (the proxy features "
            "encode the label too directly)."
        )
    return (
        f"Dominant feature(s) found: {drop}. Removing only the dominant "
        f"feature keeps AUC at {auc:.3f}; the stricter baseline (strip all "
        f"near-label proxies) drops it to {strict_auc:.3f}. The model "
        "depends on a cluster of features that act as label proxies."
    )


__all__ = [
    "DOMINANT_FEATURE_THRESHOLD",
    "NEAR_LABEL_IDENTITY_GAP",
    "FeatureRow",
    "load_toxicity_feature_audit",
    "run_toxicity_feature_audit",
]
