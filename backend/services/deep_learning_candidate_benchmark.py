from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_CSV = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_deep_learning_candidate_benchmark.json"
DEFAULT_MODEL_DIR = "Data/complete_synthetic_training/deep_learning_candidates"

BASE_NUMERIC_FEATURES = [
    "cycle",
    "age",
    "pre_wbc",
    "pre_anc",
    "pre_hemoglobin",
    "pre_platelets",
    "nadir_wbc",
    "nadir_anc",
    "nadir_hemoglobin",
    "nadir_platelets",
    "recovery_wbc",
    "recovery_hemoglobin",
    "recovery_platelets",
    "mri_tumor_size_cm",
    "mri_percent_change_from_baseline",
    "max_symptom_severity",
    "symptom_count",
    "intervention_count",
    "dose_delayed",
    "dose_reduced",
]

CATEGORICAL_FEATURES = ["stage", "molecular_subtype", "regimen"]

SYNTHETIC_GENETIC_FEATURES = [
    "genetic_record_available",
    "known_familial_mutation",
    "vus_present",
    "brca1_pathogenic_context",
    "brca2_pathogenic_context",
    "palb2_pathogenic_context",
    "tp53_pathogenic_context",
    "pten_pathogenic_context",
    "chek2_atm_pathogenic_context",
    "pik3ca_somatic_context",
    "tp53_somatic_context",
]

SYNTHETIC_TREATMENT_FEATURES = [
    "chemo_context",
    "anthracycline_taxane_context",
    "platinum_taxane_context",
    "docetaxel_carboplatin_context",
    "anti_her2_targeted_context",
    "endocrine_context",
    "radiation_planned_context",
    "surgery_planned_context",
    "immunotherapy_context",
    "parp_context",
    "supportive_growth_factor_context",
    "multi_modality_count",
]


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 24
    batch_size: int = 64
    learning_rate: float = 1e-3
    seed: int = 42
    test_size: float = 0.25
    hidden_size: int = 64


def run_deep_learning_candidate_benchmark(
    *,
    source_csv: str = DEFAULT_SOURCE_CSV,
    output_path: str = DEFAULT_OUTPUT_PATH,
    model_dir: str = DEFAULT_MODEL_DIR,
    epochs: int = 24,
    seed: int = 42,
) -> dict[str, Any]:
    config = TrainingConfig(epochs=epochs, seed=seed)
    _seed_everything(seed)

    rows = pd.read_csv(ROOT_DIR / source_csv if not Path(source_csv).is_absolute() else source_csv)
    rows = _add_synthetic_genetic_context(rows, seed=seed)
    rows = _add_synthetic_treatment_context(rows)
    prepared = _prepare_sequences(rows, config=config)

    base_features = [
        f for f in prepared["feature_names"]
        if f not in SYNTHETIC_GENETIC_FEATURES and f not in SYNTHETIC_TREATMENT_FEATURES
    ]
    variants = {
        "without_genetic_context": base_features,
        "with_genetic_context": base_features + SYNTHETIC_GENETIC_FEATURES,
        "with_treatment_context": base_features + SYNTHETIC_TREATMENT_FEATURES,
        "with_genetic_and_treatment_context": prepared["feature_names"],
    }

    results: dict[str, Any] = {}
    model_output_dir = ROOT_DIR / model_dir
    model_output_dir.mkdir(parents=True, exist_ok=True)
    for variant_name, selected_features in variants.items():
        selected_idx = [prepared["feature_names"].index(feature) for feature in selected_features]
        X_train = prepared["X_train"][:, :, selected_idx]
        X_test = prepared["X_test"][:, :, selected_idx]
        variant_results: dict[str, Any] = {}
        for model_name, factory in _model_factories(input_dim=X_train.shape[-1], hidden_size=config.hidden_size, seq_len=X_train.shape[1]).items():
            metrics, state_path = _train_and_evaluate(
                model_name=model_name,
                model_factory=factory,
                X_train=X_train,
                X_test=X_test,
                y_class_train=prepared["y_class_train"],
                y_class_test=prepared["y_class_test"],
                y_reg_train=prepared["y_reg_train"],
                y_reg_test=prepared["y_reg_test"],
                config=config,
                state_path=model_output_dir / f"{variant_name}_{model_name}.pt",
            )
            variant_results[model_name] = {**metrics, "artifact_path": str(state_path)}
        results[variant_name] = variant_results

    best = _select_best_model(results)
    best_by_task = _select_best_models_by_task(results)
    group_importance = _group_permutation_importance(
        best=best,
        prepared=prepared,
        config=config,
        model_path=ROOT_DIR / best["artifact_path"],
    )
    comparison = _compare_genetic_context(results)

    payload: dict[str, Any] = {
        "schema_version": "deep_learning_candidate_benchmark_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": _status(results, comparison),
        "source_csv": source_csv,
        "patients": int(rows["patient_id"].nunique()),
        "rows": int(len(rows)),
        "sequence_length": int(prepared["X_train"].shape[1]),
        "feature_count": {
            "base_plus_one_hot": len(variants["without_genetic_context"]),
            "with_genetic_context": len(variants["with_genetic_context"]),
            "with_treatment_context": len(variants["with_treatment_context"]),
            "with_genetic_and_treatment_context": len(variants["with_genetic_and_treatment_context"]),
            "synthetic_genetic_context": len(SYNTHETIC_GENETIC_FEATURES),
            "synthetic_treatment_context": len(SYNTHETIC_TREATMENT_FEATURES),
        },
        "targets": {
            "classification": "treatment_success_binary",
            "regression": "final-cycle response_score_percent scaled to 0..1 for training and reported as percent MAE/RMSE",
        },
        "models": results,
        "best_model": best,
        "best_models": best_by_task,
        "genetic_context_ablation": comparison,
        "treatment_context_ablation": _compare_treatment_context(results),
        "predictor_group_importance": group_importance,
        "recommended_weighting_policy": [
            "Highest evidentiary weight: direct longitudinal response evidence such as imaging trend and clinician-reviewed report summaries.",
            "High monitoring weight: CBC/lab trajectory, symptoms, treatment-cycle timing, and dose/intervention events.",
            "Contextual weight: ER/PR/HER2/Ki-67 and known treatment regimen context.",
            "Referral/context weight: germline genetic-test records and family-history readiness; useful for review routing, not standalone response proof.",
            "Lowest standalone weight: tumor-marker trends; review context only, never recurrence/progression proof by themselves.",
        ],
        "genetic_feature_boundary": (
            "Synthetic genetic mutation features are included only as contextual candidate predictors. "
            "They are not inferred from imaging, not interpreted as medical advice, and not promoted "
            "without real genetic-test records plus external validation."
        ),
        "treatment_feature_boundary": (
            "Synthetic treatment-combination features encode regimen/timeline context only. They are not treatment "
            "recommendations, do not compare real treatment efficacy, and cannot be used to tell a patient to start, "
            "stop, delay, or switch therapy."
        ),
        "claim_boundary": (
            "Deep-learning results are synthetic engineering baselines. They measure ability to learn the simulator, "
            "not real patient response, prognosis, genetic risk, or clinical treatment utility."
        ),
    }

    output = ROOT_DIR / output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _add_synthetic_genetic_context(rows: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    patient_rows: list[dict[str, Any]] = []
    for patient_id, group in rows.groupby("patient_id", sort=True):
        first = group.iloc[0]
        subtype = str(first.get("molecular_subtype", ""))
        stage = str(first.get("stage", ""))
        age = float(first.get("age", np.nan))
        triple_negative = subtype == "triple-negative"
        her2_pos = "HER2+" in subtype
        advanced = stage in {"IIIA", "IIIB", "IIIC", "IV"}
        young = not math.isnan(age) and age < 50

        record_available = rng.random() < (0.72 if young or triple_negative or advanced else 0.48)
        brca1 = record_available and rng.random() < (0.12 if triple_negative else 0.035)
        brca2 = record_available and rng.random() < (0.055 if not triple_negative else 0.035)
        palb2 = record_available and rng.random() < 0.025
        tp53_germline = record_available and rng.random() < (0.018 if young else 0.006)
        pten = record_available and rng.random() < 0.006
        chek2_atm = record_available and rng.random() < (0.04 if not triple_negative else 0.02)
        known_familial = any([brca1, brca2, palb2, tp53_germline, pten, chek2_atm]) and rng.random() < 0.62
        vus = record_available and rng.random() < 0.16
        pik3ca_somatic = rng.random() < (0.28 if "HR+" in subtype else 0.08)
        tp53_somatic = rng.random() < (0.44 if triple_negative or her2_pos else 0.18)
        patient_rows.append({
            "patient_id": patient_id,
            "genetic_record_available": int(record_available),
            "known_familial_mutation": int(known_familial),
            "vus_present": int(vus),
            "brca1_pathogenic_context": int(brca1),
            "brca2_pathogenic_context": int(brca2),
            "palb2_pathogenic_context": int(palb2),
            "tp53_pathogenic_context": int(tp53_germline),
            "pten_pathogenic_context": int(pten),
            "chek2_atm_pathogenic_context": int(chek2_atm),
            "pik3ca_somatic_context": int(pik3ca_somatic),
            "tp53_somatic_context": int(tp53_somatic),
        })
    return rows.merge(pd.DataFrame(patient_rows), on="patient_id", how="left")


def _add_synthetic_treatment_context(rows: pd.DataFrame) -> pd.DataFrame:
    enriched = rows.copy()
    lower_regimen = enriched["regimen"].fillna("").astype(str).str.lower()
    subtype = enriched["molecular_subtype"].fillna("").astype(str)
    stage = enriched["stage"].fillna("").astype(str)
    cycle = pd.to_numeric(enriched["cycle"], errors="coerce").fillna(0)

    enriched["chemo_context"] = 1
    enriched["anthracycline_taxane_context"] = lower_regimen.str.contains("ac|taxane|paclitaxel|dose-dense").astype(int)
    enriched["platinum_taxane_context"] = lower_regimen.str.contains("carboplatin|platinum").astype(int)
    enriched["docetaxel_carboplatin_context"] = lower_regimen.str.contains("tchp|docetaxel").astype(int)
    enriched["anti_her2_targeted_context"] = (
        subtype.str.contains("HER2\\+", regex=True) | lower_regimen.str.contains("tchp|trastuzumab|pertuzumab")
    ).astype(int)
    enriched["endocrine_context"] = (subtype.str.contains("HR\\+", regex=True) & (cycle >= 5)).astype(int)
    enriched["radiation_planned_context"] = (stage.isin(["IIA", "IIB", "IIIA", "IIIB", "IIIC", "IV"]) & (cycle >= 5)).astype(int)
    enriched["surgery_planned_context"] = (cycle >= 6).astype(int)
    enriched["immunotherapy_context"] = (
        subtype.eq("triple-negative") & lower_regimen.str.contains("carboplatin|paclitaxel")
    ).astype(int)
    enriched["parp_context"] = (
        subtype.eq("triple-negative") & stage.isin(["IIIA", "IIIB", "IIIC", "IV"]) & (cycle >= 5)
    ).astype(int)
    enriched["supportive_growth_factor_context"] = (
        (pd.to_numeric(enriched.get("nadir_anc"), errors="coerce") < 1.2)
        | (pd.to_numeric(enriched.get("dose_delayed"), errors="coerce").fillna(0) > 0)
    ).astype(int)
    modality_cols = [
        "chemo_context",
        "anti_her2_targeted_context",
        "endocrine_context",
        "radiation_planned_context",
        "surgery_planned_context",
        "immunotherapy_context",
        "parp_context",
    ]
    enriched["multi_modality_count"] = enriched[modality_cols].sum(axis=1)
    return enriched


def _prepare_sequences(rows: pd.DataFrame, *, config: TrainingConfig) -> dict[str, Any]:
    rows = rows.sort_values(["patient_id", "cycle"]).copy()
    rows[CATEGORICAL_FEATURES] = rows[CATEGORICAL_FEATURES].fillna("unknown").astype(str)
    encoded = pd.get_dummies(rows[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES, dtype=float)
    context_features = SYNTHETIC_GENETIC_FEATURES + SYNTHETIC_TREATMENT_FEATURES
    model_rows = pd.concat([rows[["patient_id", "treatment_success_binary", "response_score_percent"]], rows[BASE_NUMERIC_FEATURES + context_features], encoded], axis=1)
    feature_names = BASE_NUMERIC_FEATURES + context_features + list(encoded.columns)

    patient_targets = rows.groupby("patient_id").agg(
        treatment_success_binary=("treatment_success_binary", "first"),
        response_score_percent=("response_score_percent", "last"),
    )
    patients = patient_targets.index.to_numpy()
    y_class = patient_targets["treatment_success_binary"].astype(int).to_numpy()
    train_patients, test_patients = train_test_split(
        patients,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=y_class,
    )
    train_set = set(train_patients)
    train_rows = model_rows[model_rows["patient_id"].isin(train_set)].copy()
    scaler = StandardScaler()
    scaler.fit(train_rows[feature_names].fillna(0.0).astype(float))
    model_rows[feature_names] = scaler.transform(model_rows[feature_names].fillna(0.0).astype(float))

    X_train, yc_train, yr_train, train_ids = _patient_tensor(model_rows, feature_names, train_patients)
    X_test, yc_test, yr_test, test_ids = _patient_tensor(model_rows, feature_names, test_patients)
    return {
        "feature_names": feature_names,
        "X_train": X_train,
        "X_test": X_test,
        "y_class_train": yc_train,
        "y_class_test": yc_test,
        "y_reg_train": yr_train,
        "y_reg_test": yr_test,
        "train_patient_ids": train_ids,
        "test_patient_ids": test_ids,
        "patient_split_disjoint": not bool(set(train_ids) & set(test_ids)),
    }


def _patient_tensor(model_rows: pd.DataFrame, feature_names: list[str], patient_ids: np.ndarray):
    sequences: list[np.ndarray] = []
    y_class: list[float] = []
    y_reg: list[float] = []
    kept_ids: list[str] = []
    max_len = int(model_rows.groupby("patient_id").size().max())
    for patient_id in patient_ids:
        group = model_rows[model_rows["patient_id"] == patient_id].sort_values("cycle")
        if group.empty:
            continue
        seq = group[feature_names].astype(float).to_numpy(dtype=np.float32)
        if len(seq) < max_len:
            pad = np.zeros((max_len - len(seq), seq.shape[1]), dtype=np.float32)
            seq = np.vstack([seq, pad])
        sequences.append(seq[:max_len])
        y_class.append(float(group["treatment_success_binary"].iloc[0]))
        y_reg.append(float(group["response_score_percent"].iloc[-1]) / 100.0)
        kept_ids.append(str(patient_id))
    return (
        np.stack(sequences).astype(np.float32),
        np.asarray(y_class, dtype=np.float32),
        np.asarray(y_reg, dtype=np.float32),
        kept_ids,
    )


def _model_factories(*, input_dim: int, hidden_size: int, seq_len: int):
    return {
        "sequence_mlp": lambda: SequenceMLP(input_dim=input_dim, hidden_size=hidden_size, seq_len=seq_len),
        "temporal_cnn": lambda: TemporalCNN(input_dim=input_dim, hidden_size=hidden_size),
        "bidirectional_gru": lambda: BiGRU(input_dim=input_dim, hidden_size=hidden_size),
        "tiny_transformer": lambda: TinyTransformer(input_dim=input_dim, hidden_size=hidden_size, seq_len=seq_len),
    }


def _train_and_evaluate(
    *,
    model_name: str,
    model_factory,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_class_train: np.ndarray,
    y_class_test: np.ndarray,
    y_reg_train: np.ndarray,
    y_reg_test: np.ndarray,
    config: TrainingConfig,
    state_path: Path,
) -> tuple[dict[str, Any], Path]:
    model = model_factory()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_class_train).unsqueeze(1),
        torch.from_numpy(y_reg_train).unsqueeze(1),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model.train()
    for _ in range(config.epochs):
        for xb, ycb, yrb in loader:
            optimizer.zero_grad()
            class_logits, reg = model(xb)
            loss = bce(class_logits, ycb) + mse(reg, yrb)
            loss.backward()
            optimizer.step()

    metrics = _evaluate_model(model, X_test, y_class_test, y_reg_test)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": model_name, "state_dict": model.state_dict(), "input_dim": X_train.shape[-1]}, state_path)
    return metrics, state_path


def _evaluate_model(model: nn.Module, X: np.ndarray, y_class: np.ndarray, y_reg: np.ndarray) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        class_logits, reg = model(torch.from_numpy(X))
        probs = torch.sigmoid(class_logits).squeeze(1).cpu().numpy()
        reg_pred = torch.clamp(reg.squeeze(1), 0.0, 1.0).cpu().numpy()
    y_pred = (probs >= 0.5).astype(int)
    y_reg_pct = y_reg * 100.0
    reg_pred_pct = reg_pred * 100.0
    return {
        "classification": {
            "auroc": _safe_auc(y_class, probs),
            "pr_auc": _safe_average_precision(y_class, probs),
            "brier": float(brier_score_loss(y_class, probs)),
            "accuracy": float(accuracy_score(y_class, y_pred)),
        },
        "regression": {
            "mae_percent": float(mean_absolute_error(y_reg_pct, reg_pred_pct)),
            "rmse_percent": float(np.sqrt(mean_squared_error(y_reg_pct, reg_pred_pct))),
            "r2": float(r2_score(y_reg_pct, reg_pred_pct)),
        },
    }


def _group_permutation_importance(
    *,
    best: dict[str, Any],
    prepared: dict[str, Any],
    config: TrainingConfig,
    model_path: Path,
) -> list[dict[str, Any]]:
    selected_features = _features_for_variant(prepared["feature_names"], best["variant"])
    selected_idx = [prepared["feature_names"].index(feature) for feature in selected_features]
    X = prepared["X_test"][:, :, selected_idx].copy()
    y_class = prepared["y_class_test"]
    y_reg = prepared["y_reg_test"]
    model = _model_factories(input_dim=X.shape[-1], hidden_size=config.hidden_size, seq_len=X.shape[1])[best["model"]]()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    base = _evaluate_model(model, X, y_class, y_reg)
    rng = np.random.default_rng(config.seed + 99)
    groups = _feature_groups(selected_features)
    rows: list[dict[str, Any]] = []
    for group_name, group_features in groups.items():
        idx = [selected_features.index(feature) for feature in group_features if feature in selected_features]
        if not idx:
            continue
        X_perm = X.copy()
        perm = rng.permutation(X_perm.shape[0])
        X_perm[:, :, idx] = X_perm[perm][:, :, idx]
        perturbed = _evaluate_model(model, X_perm, y_class, y_reg)
        rows.append({
            "group": group_name,
            "classification_auroc_drop": _round_or_none(
                (base["classification"]["auroc"] or 0.0) - (perturbed["classification"]["auroc"] or 0.0)
            ),
            "regression_mae_increase": _round_or_none(
                perturbed["regression"]["mae_percent"] - base["regression"]["mae_percent"]
            ),
            "recommended_role": _recommended_role(group_name),
        })
    return sorted(
        rows,
        key=lambda row: (
            row["classification_auroc_drop"] or 0.0,
            row["regression_mae_increase"] or 0.0,
        ),
        reverse=True,
    )


def _feature_groups(feature_names: list[str]) -> dict[str, list[str]]:
    return {
        "imaging_response": [f for f in feature_names if f.startswith("mri_")],
        "cbc_labs": [f for f in feature_names if any(token in f for token in ("wbc", "anc", "hemoglobin", "platelets"))],
        "symptoms_interventions": [
            f for f in feature_names if f in {"max_symptom_severity", "symptom_count", "intervention_count", "dose_delayed", "dose_reduced"}
        ],
        "treatment_context": [f for f in feature_names if f.startswith("regimen_") or f == "cycle"],
        "demographics_stage": [f for f in feature_names if f == "age" or f.startswith("stage_")],
        "biomarker_subtype": [f for f in feature_names if f.startswith("molecular_subtype_")],
        "synthetic_genetic_context": [f for f in feature_names if f in SYNTHETIC_GENETIC_FEATURES],
        "synthetic_treatment_context": [f for f in feature_names if f in SYNTHETIC_TREATMENT_FEATURES],
    }


def _recommended_role(group_name: str) -> str:
    return {
        "imaging_response": "highest direct response-monitoring signal when report timing is valid",
        "cbc_labs": "high toxicity/monitoring signal; response support only",
        "symptoms_interventions": "high review-routing signal; patient-reported and context-dependent",
        "treatment_context": "context and temporal alignment signal",
        "demographics_stage": "baseline context; should not dominate",
        "biomarker_subtype": "contextual modifier for expected treatment pathways",
        "synthetic_genetic_context": "genetic-counselor review and contextual modifier only",
        "synthetic_treatment_context": "treatment-combination context only, never treatment recommendation",
    }.get(group_name, "context")


def _select_best_model(results: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for variant, models in results.items():
        for model_name, metrics in models.items():
            rows.append({
                "variant": variant,
                "model": model_name,
                "artifact_path": metrics["artifact_path"],
                "classification_auroc": metrics["classification"]["auroc"],
                "classification_brier": metrics["classification"]["brier"],
                "regression_mae_percent": metrics["regression"]["mae_percent"],
                "regression_r2": metrics["regression"]["r2"],
            })
    return sorted(
        rows,
        key=lambda row: (
            row["classification_auroc"] or 0.0,
            -(row["classification_brier"] or 1.0),
            -(row["regression_mae_percent"] or 999.0),
        ),
        reverse=True,
    )[0]


def _select_best_models_by_task(results: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for variant, models in results.items():
        for model_name, metrics in models.items():
            rows.append({
                "variant": variant,
                "model": model_name,
                "artifact_path": metrics["artifact_path"],
                "classification_auroc": metrics["classification"]["auroc"],
                "classification_brier": metrics["classification"]["brier"],
                "regression_mae_percent": metrics["regression"]["mae_percent"],
                "regression_r2": metrics["regression"]["r2"],
            })
    return {
        "classification": sorted(
            rows,
            key=lambda row: (
                row["classification_auroc"] or 0.0,
                -(row["classification_brier"] or 1.0),
            ),
            reverse=True,
        )[0],
        "regression": sorted(
            rows,
            key=lambda row: (
                -(row["regression_mae_percent"] or 999.0),
                row["regression_r2"],
            ),
            reverse=True,
        )[0],
    }


def _features_for_variant(feature_names: list[str], variant: str) -> list[str]:
    base = [f for f in feature_names if f not in SYNTHETIC_GENETIC_FEATURES and f not in SYNTHETIC_TREATMENT_FEATURES]
    if variant == "without_genetic_context":
        return base
    if variant == "with_genetic_context":
        return base + SYNTHETIC_GENETIC_FEATURES
    if variant == "with_treatment_context":
        return base + SYNTHETIC_TREATMENT_FEATURES
    return feature_names


def _compare_genetic_context(results: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for model_name, with_metrics in results["with_genetic_context"].items():
        without_metrics = results["without_genetic_context"][model_name]
        rows.append({
            "model": model_name,
            "classification_auroc_delta": _round_or_none(
                (with_metrics["classification"]["auroc"] or 0.0) - (without_metrics["classification"]["auroc"] or 0.0)
            ),
            "classification_brier_delta": _round_or_none(
                with_metrics["classification"]["brier"] - without_metrics["classification"]["brier"]
            ),
            "regression_mae_delta": _round_or_none(
                with_metrics["regression"]["mae_percent"] - without_metrics["regression"]["mae_percent"]
            ),
            "regression_r2_delta": _round_or_none(
                with_metrics["regression"]["r2"] - without_metrics["regression"]["r2"]
            ),
        })
    best_classification_delta = max(row["classification_auroc_delta"] or 0.0 for row in rows)
    best_regression_delta = min(row["regression_mae_delta"] or 0.0 for row in rows)
    return {
        "rows": rows,
        "best_classification_auroc_delta": best_classification_delta,
        "best_regression_mae_delta": best_regression_delta,
        "decision": (
            "context_only_no_promotion"
            if best_classification_delta < 0.01 and best_regression_delta > -1.0
            else "candidate_for_external_validation_only"
        ),
        "interpretation": (
            "Synthetic genetic context can be benchmarked, but any gain must be treated as simulator-specific "
            "until real genetic-test records and external validation exist."
        ),
    }


def _compare_treatment_context(results: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for model_name, with_metrics in results["with_treatment_context"].items():
        without_metrics = results["without_genetic_context"][model_name]
        rows.append({
            "model": model_name,
            "classification_auroc_delta": _round_or_none(
                (with_metrics["classification"]["auroc"] or 0.0) - (without_metrics["classification"]["auroc"] or 0.0)
            ),
            "classification_brier_delta": _round_or_none(
                with_metrics["classification"]["brier"] - without_metrics["classification"]["brier"]
            ),
            "regression_mae_delta": _round_or_none(
                with_metrics["regression"]["mae_percent"] - without_metrics["regression"]["mae_percent"]
            ),
            "regression_r2_delta": _round_or_none(
                with_metrics["regression"]["r2"] - without_metrics["regression"]["r2"]
            ),
        })
    best_classification_delta = max(row["classification_auroc_delta"] or 0.0 for row in rows)
    best_regression_delta = min(row["regression_mae_delta"] or 0.0 for row in rows)
    return {
        "rows": rows,
        "best_classification_auroc_delta": best_classification_delta,
        "best_regression_mae_delta": best_regression_delta,
        "decision": "context_only_no_treatment_recommendation",
        "interpretation": (
            "Synthetic treatment-combination features can improve or stabilize offline signals, but they encode "
            "treatment context only. They must not be used to recommend a regimen or compare real-world efficacy."
        ),
    }


def _status(results: dict[str, Any], comparison: dict[str, Any]) -> str:
    best_auroc = max(
        metrics["classification"]["auroc"] or 0.0
        for models in results.values()
        for metrics in models.values()
    )
    best_r2 = max(
        metrics["regression"]["r2"]
        for models in results.values()
        for metrics in models.values()
    )
    if best_auroc >= 0.85 and best_r2 >= 0.5 and comparison["decision"] in {"context_only_no_promotion", "candidate_for_external_validation_only"}:
        return "strong"
    if best_auroc >= 0.75 and best_r2 >= 0.2:
        return "acceptable"
    return "needs_attention"


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(set(y_true.astype(int).tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(set(y_true.astype(int).tolist())) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


class SequenceMLP(nn.Module):
    def __init__(self, *, input_dim: int, hidden_size: int, seq_len: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * seq_len, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
        )
        self.class_head = nn.Linear(hidden_size, 1)
        self.reg_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, x):
        z = self.net(x)
        return self.class_head(z), self.reg_head(z)


class TemporalCNN(nn.Module):
    def __init__(self, *, input_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.class_head = nn.Linear(hidden_size, 1)
        self.reg_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, x):
        z = self.net(x.transpose(1, 2)).squeeze(-1)
        return self.class_head(z), self.reg_head(z)


class BiGRU(nn.Module):
    def __init__(self, *, input_dim: int, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.15)
        self.class_head = nn.Linear(hidden_size * 2, 1)
        self.reg_head = nn.Sequential(nn.Linear(hidden_size * 2, 1), nn.Sigmoid())

    def forward(self, x):
        _, hidden = self.gru(x)
        z = torch.cat([hidden[-2], hidden[-1]], dim=1)
        z = self.dropout(z)
        return self.class_head(z), self.reg_head(z)


class TinyTransformer(nn.Module):
    def __init__(self, *, input_dim: int, hidden_size: int, seq_len: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_size)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.class_head = nn.Linear(hidden_size, 1)
        self.reg_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, x):
        z = self.proj(x) + self.pos
        z = self.encoder(z).mean(dim=1)
        return self.class_head(z), self.reg_head(z)
