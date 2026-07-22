"""Leakage-aware Duke Breast Cancer MRI tabular stress benchmark.

The public cohort's neoadjuvant pathologic-response endpoint is a different
task from NLCare's synthetic longitudinal monitoring heads. Nothing produced
here is used by patient-facing inference or promoted into the NLCare registry.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
BRIDGE_DIR = ROOT / "Data" / "external_bridge" / "duke_tcia"
CLINICAL_PATH = BRIDGE_DIR / "Clinical_and_Other_Features.xlsx"
MRI_PATH = BRIDGE_DIR / "Imaging_Features.xlsx"
CANONICAL_PATH = BRIDGE_DIR / "canonical_duke_tcia_tabular.csv"
OUTPUT_PATH = ROOT / "Data" / "evals" / "models" / "latest_duke_tcia_external_stress.json"

EXPECTED_SHA256 = {
    CLINICAL_PATH.name: "8ef0945c9f7513acd2d9c6e8866d98f2234dd78443110d9496c24a82dbbc7e6f",
    MRI_PATH.name: "371e7aec937ccf7cff01bc3c66063ce22d9699d3d3f8c6ac0dcee6728826682d",
}
OFFICIAL_COLLECTION_URL = "https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/"
SEEDS = (19, 31, 47, 73, 109)
MRI_FEATURE_COUNT = 32
TARGET_SOURCE = "Pathologic Response to Neoadjuvant Therapy"

CLINICAL_COLUMN_MAP = {
    "Date of Birth (Days)": "age_at_diagnosis_years",
    "Menopause (at diagnosis)": "menopause_at_diagnosis",
    "Metastatic at Presentation (Outside of Lymph Nodes)": "metastatic_at_presentation",
    "ER": "er",
    "PR": "pr",
    "HER2": "her2",
    "Mol Subtype": "molecular_subtype",
    "Staging(Tumor Size)# [T]": "clinical_stage_t",
    "Staging(Nodes)#(Nx replaced by -1)[N]": "clinical_stage_n",
    "Staging(Metastasis)#(Mx -replaced by -1)[M]": "clinical_stage_m",
    "Nottingham grade": "nottingham_grade",
    "Multicentric/Multifocal": "multicentric_multifocal",
    "Contralateral Breast Involvement": "contralateral_involvement",
    "Lymphadenopathy or Suspicious Nodes": "suspicious_nodes",
    "Skin/Nipple Invovlement": "skin_nipple_involvement",
    "Pec/Chest Involvement": "chest_involvement",
}
CLINICAL_FEATURES = list(CLINICAL_COLUMN_MAP.values())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_inputs(clinical_path: Path, mri_path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in (clinical_path, mri_path):
        if not path.exists():
            raise FileNotFoundError(f"Required official Duke/TCIA file is missing: {path}")
        actual = _sha256(path)
        expected = EXPECTED_SHA256.get(path.name)
        if expected is None or actual != expected:
            raise ValueError(f"Checksum mismatch for {path.name}: {actual}")
        hashes[path.name] = actual
    return hashes


def _case_key(value: Any) -> str:
    return hashlib.sha256(f"duke-tcia:{value}".encode("utf-8")).hexdigest()[:16]


def _require_columns(frame: pd.DataFrame, expected: set[str], label: str) -> None:
    missing = sorted(expected - set(frame.columns))
    if missing:
        raise ValueError(f"{label} schema missing required columns: {missing}")


def _select_mri_columns(frame: pd.DataFrame) -> list[str]:
    selected: list[str] = []
    for column in frame.columns:
        if column == "Patient ID":
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().mean() < 0.80 or numeric.nunique(dropna=True) < 2:
            continue
        selected.append(str(column))
        if len(selected) == MRI_FEATURE_COUNT:
            break
    if len(selected) < MRI_FEATURE_COUNT:
        raise ValueError(f"Only {len(selected)} usable MRI columns; expected {MRI_FEATURE_COUNT}")
    return selected


def canonicalize_frames(
    clinical: pd.DataFrame,
    mri: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_clinical = {"Patient ID", TARGET_SOURCE, *CLINICAL_COLUMN_MAP.keys()}
    _require_columns(clinical, required_clinical, "clinical")
    _require_columns(mri, {"Patient ID"}, "MRI")
    if clinical["Patient ID"].duplicated().any() or mri["Patient ID"].duplicated().any():
        raise ValueError("Duke/TCIA subject identifiers must be one-to-one before joining")

    mri_columns = _select_mri_columns(mri)
    joined = clinical.merge(mri[["Patient ID", *mri_columns]], on="Patient ID", validate="one_to_one")
    response = pd.to_numeric(joined[TARGET_SOURCE], errors="coerce")
    eligible = response.notna()
    joined = joined.loc[eligible].reset_index(drop=True)
    response = response.loc[eligible].reset_index(drop=True)

    canonical = pd.DataFrame({
        "external_case_key": joined["Patient ID"].map(_case_key),
        "external_target_pathologic_complete_response": (response == 1).astype(int),
    })
    for source, target in CLINICAL_COLUMN_MAP.items():
        values = pd.to_numeric(joined[source], errors="coerce")
        if source == "Date of Birth (Days)":
            values = -values / 365.25
        canonical[target] = values
    mri_map: dict[str, str] = {}
    for index, source in enumerate(mri_columns, start=1):
        target = f"mri_feature_{index:03d}"
        canonical[target] = pd.to_numeric(joined[source], errors="coerce")
        mri_map[target] = source

    manifest = {
        "clinical_row_count": int(len(clinical)),
        "mri_row_count": int(len(mri)),
        "joined_labeled_row_count": int(len(canonical)),
        "positive_target_n": int(canonical["external_target_pathologic_complete_response"].sum()),
        "raw_subject_identifier_exported": False,
        "treatment_columns_exported_or_used_as_features": False,
        "recurrence_or_survival_columns_used_as_features": False,
        "mri_feature_selection": "first 32 non-constant numeric columns with at least 80% observed values; target-blind",
        "mri_feature_source_map": mri_map,
    }
    return canonical, manifest


def build_canonical_frame(
    clinical_path: Path = CLINICAL_PATH,
    mri_path: Path = MRI_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    hashes = _verify_inputs(clinical_path, mri_path)
    clinical = pd.read_excel(clinical_path, sheet_name="Data", header=1, skiprows=[2])
    mri = pd.read_excel(mri_path, sheet_name="Imaging Features")
    canonical, manifest = canonicalize_frames(clinical, mri)
    return canonical, {"input_sha256": hashes, **manifest}


def _models(seed: int) -> dict[str, Pipeline]:
    return {
        "prevalence_dummy": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", DummyClassifier(strategy="prior")),
        ]),
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=seed)),
        ]),
        "gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", GradientBoostingClassifier(random_state=seed, n_estimators=100, max_depth=2)),
        ]),
    }


def _ece(y_true: pd.Series, probability: np.ndarray, bins: int = 8) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    score = 0.0
    values = np.asarray(y_true, dtype=float)
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (probability < upper if upper < 1.0 else probability <= upper)
        if not mask.any():
            continue
        score += (mask.sum() / total) * abs(float(probability[mask].mean()) - float(values[mask].mean()))
    return float(score)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for feature_set, model in sorted({(row["feature_set"], row["model"]) for row in rows}):
        group = [row for row in rows if row["feature_set"] == feature_set and row["model"] == model]
        item: dict[str, Any] = {"feature_set": feature_set, "model": model, "n_seeds": len(group)}
        for metric in ("auroc", "average_precision", "brier", "ece"):
            values = np.asarray([row[metric] for row in group], dtype=float)
            item[f"mean_{metric}"] = round(float(values.mean()), 6)
            item[f"min_{metric}"] = round(float(values.min()), 6)
            item[f"max_{metric}"] = round(float(values.max()), 6)
        output.append(item)
    return output


def _bootstrap_mean_ci(values: list[float], seed: int = 20260722) -> list[float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(array, size=len(array), replace=True).mean()) for _ in range(2000)]
    return [round(float(np.quantile(means, 0.025)), 6), round(float(np.quantile(means, 0.975)), 6)]


def _paired_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for model in ("logistic_regression", "gradient_boosting"):
        clinical = {row["seed"]: row for row in rows if row["model"] == model and row["feature_set"] == "clinical_only"}
        combined = {row["seed"]: row for row in rows if row["model"] == model and row["feature_set"] == "clinical_plus_mri"}
        for metric in ("auroc", "average_precision", "brier", "ece"):
            deltas = [float(combined[seed][metric]) - float(clinical[seed][metric]) for seed in SEEDS]
            output.append({
                "model": model,
                "comparison": "clinical_plus_mri_minus_clinical_only",
                "metric": metric,
                "mean_delta": round(float(np.mean(deltas)), 6),
                "bootstrap_95_ci": _bootstrap_mean_ci(deltas),
                "n_paired_seeds": len(deltas),
                "direction_note": "lower_is_better" if metric in {"brier", "ece"} else "higher_is_better",
            })
    return output


def run_duke_tcia_external_stress(
    clinical_path: Path = CLINICAL_PATH,
    mri_path: Path = MRI_PATH,
    canonical_path: Path = CANONICAL_PATH,
    output_path: Path = OUTPUT_PATH,
) -> dict[str, Any]:
    canonical, source_manifest = build_canonical_frame(clinical_path, mri_path)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(canonical_path, index=False)

    target_name = "external_target_pathologic_complete_response"
    target = canonical[target_name]
    mri_features = [column for column in canonical if column.startswith("mri_feature_")]
    feature_sets = {
        "clinical_only": CLINICAL_FEATURES,
        "mri_only": mri_features,
        "clinical_plus_mri": [*CLINICAL_FEATURES, *mri_features],
    }
    rows: list[dict[str, Any]] = []
    indices = np.arange(len(canonical))
    for seed in SEEDS:
        train_idx, test_idx = train_test_split(indices, test_size=0.30, random_state=seed, stratify=target)
        for feature_set, columns in feature_sets.items():
            x_train = canonical.loc[train_idx, columns]
            x_test = canonical.loc[test_idx, columns]
            y_train = target.iloc[train_idx]
            y_test = target.iloc[test_idx]
            for model_name, model in _models(seed).items():
                model.fit(x_train, y_train)
                probability = model.predict_proba(x_test)[:, 1]
                rows.append({
                    "seed": seed,
                    "feature_set": feature_set,
                    "model": model_name,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "test_prevalence": round(float(y_test.mean()), 6),
                    "auroc": round(float(roc_auc_score(y_test, probability)), 6),
                    "average_precision": round(float(average_precision_score(y_test, probability)), 6),
                    "brier": round(float(brier_score_loss(y_test, probability)), 6),
                    "ece": round(_ece(y_test, probability), 6),
                })

    payload = {
        "schema_version": "duke_tcia_external_stress_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "acceptable_external_engineering_stress",
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "used_for_nlcare_training": False,
        "patient_facing_allowed": False,
        "promotion_allowed": False,
        "source": {
            "dataset": "Duke Breast Cancer MRI",
            "publisher": "The Cancer Imaging Archive",
            "official_collection_url": OFFICIAL_COLLECTION_URL,
            "license": "CC BY-NC 4.0 as stated by the official collection page",
            **source_manifest,
        },
        "canonical_export": {
            "path": canonical_path.relative_to(ROOT).as_posix() if canonical_path.is_relative_to(ROOT) else str(canonical_path),
            "sha256": _sha256(canonical_path),
            "row_count": int(len(canonical)),
            "raw_subject_identifier_exported": False,
            "treatment_columns_exported_or_used_as_features": False,
        },
        "task_boundary": {
            "external_target": "coded pathologic complete response after neoadjuvant therapy",
            "nlcare_target_match": False,
            "reason": "A cohort-level neoadjuvant response endpoint is not the same task as NLCare's synthetic monitoring heads.",
            "allowed_reading": "separate public-data ingestion, leakage control, baseline, calibration, and uncertainty stress test",
        },
        "protocol": {
            "seeds": list(SEEDS),
            "split": "repeated stratified 70/30 holdout",
            "feature_sets": {key: value for key, value in feature_sets.items()},
            "models": ["prevalence_dummy", "logistic_regression", "gradient_boosting"],
            "treatment_recurrence_survival_features_used": False,
        },
        "per_seed_results": rows,
        "aggregate_results": _aggregate(rows),
        "paired_feature_deltas": _paired_deltas(rows),
        "claim_boundary": (
            "Public-data engineering stress evidence only. This is not clinical validation, "
            "not an NLCare model promotion result, and not evidence of patient benefit or safety."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


__all__ = ["canonicalize_frames", "build_canonical_frame", "run_duke_tcia_external_stress"]
