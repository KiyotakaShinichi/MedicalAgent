from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_public_biomarker_dataset_readiness.json"
BIOMARKER_BENCHMARK_PATH = ROOT_DIR / "Data" / "mle_monitoring" / "biomarker_feature_benchmark.json"
FULL_ABLATION_PATH = ROOT_DIR / "Data" / "evals" / "models" / "latest_full_feature_group_ablation.json"


DATASET_CANDIDATES: list[dict[str, Any]] = [
    {
        "id": "breastdcedl",
        "name": "BreastDCEDL DCE-MRI treatment-response dataset",
        "source_url": "https://zenodo.org/records/17274053",
        "secondary_source_url": "https://www.nature.com/articles/s41597-026-06589-6",
        "access": "public non-commercial research dataset",
        "sample_count": 2070,
        "biomarker_fields": ["HR status", "HER2 status"],
        "tumor_marker_fields": [],
        "imaging_fields": ["3D DCE-MRI", "tumor segmentation"],
        "outcomes": ["pathologic complete response"],
        "readiness": "mapped_locally_external_benchmark",
        "allowed_use": [
            "external imaging plus HR/HER2 benchmark",
            "pCR response modeling experiment",
            "synthetic-to-public feature alignment",
        ],
        "limitations": [
            "No CBC, symptom, medication, or longitudinal treatment-cycle timeline.",
            "No CA 15-3, CA 27.29, or CEA serum tumor-marker trajectory.",
            "Non-commercial derivative license must be respected.",
        ],
    },
    {
        "id": "metabric_cbioportal",
        "name": "METABRIC breast cancer cohort via cBioPortal",
        "source_url": "https://www.cbioportal.org/api/studies/brca_metabric?projection=SUMMARY",
        "secondary_source_url": "https://datacatalog.mskcc.org/dataset/11457",
        "access": "public cBioPortal/API where permitted",
        "sample_count": 2509,
        "biomarker_fields": ["ER status", "PR status", "HER2 status", "PAM50/subtype", "grade", "stage"],
        "tumor_marker_fields": [],
        "omics_fields": ["targeted sequencing", "copy number", "mRNA microarray"],
        "outcomes": ["survival/outcome fields"],
        "readiness": "api_schema_ready_biomarker_external_check",
        "allowed_use": [
            "subtype/genomic external schema check",
            "biomarker distribution alignment",
            "non-longitudinal survival sanity check",
        ],
        "limitations": [
            "Not an NLCare-style CBC/symptom/imaging-report timeline.",
            "Not a public serum tumor-marker response cohort.",
        ],
    },
    {
        "id": "tcga_brca_pan_can_atlas",
        "name": "TCGA-BRCA PanCancer Atlas via cBioPortal/GDC",
        "source_url": "https://www.cbioportal.org/api/studies/brca_tcga_pan_can_atlas_2018?projection=SUMMARY",
        "secondary_source_url": "https://api.gdc.cancer.gov/projects/TCGA-BRCA",
        "access": "public GDC/cBioPortal APIs for open data; controlled files require token",
        "sample_count": 1084,
        "biomarker_fields": ["subtype", "stage", "grade", "genomic alteration fields", "protein/RPPA where available"],
        "tumor_marker_fields": [],
        "omics_fields": ["DNA", "RNA", "copy number", "RPPA/proteomics where available"],
        "outcomes": ["OS", "DFS", "PFS where available"],
        "readiness": "api_schema_ready_biomarker_external_check",
        "allowed_use": [
            "genomic/subtype external schema check",
            "distribution comparison",
            "model-card external-readiness evidence",
        ],
        "limitations": [
            "Open TCGA/GDC data are not a serial monitoring workflow.",
            "Controlled data access may require authorization.",
            "Serum CA 15-3, CA 27.29, and CEA trends are not the core public signal.",
        ],
    },
    {
        "id": "cptac_breast_proteogenomic",
        "name": "CPTAC breast cancer proteogenomic resources",
        "source_url": "https://gdc.cancer.gov/about-gdc/contributed-genomic-data-cancer-research/clinical-proteomic-tumor-analysis-consortium-cptac",
        "secondary_source_url": "https://dctd.cancer.gov/data-tools-biospecimens/data/pdc",
        "access": "public PDC/GDC resources with source-specific access rules",
        "sample_count": None,
        "biomarker_fields": ["ER", "PR", "HER2", "PAM50/subtype", "proteomics/phosphoproteomics"],
        "tumor_marker_fields": [],
        "omics_fields": ["proteomics", "phosphoproteomics", "genomics"],
        "outcomes": ["clinical annotations; not NLCare treatment-cycle response"],
        "readiness": "manual_download_external_proteomics_candidate",
        "allowed_use": [
            "future proteogenomic feature sanity check",
            "biomarker biology context",
            "distribution alignment after manual download",
        ],
        "limitations": [
            "Requires manual data retrieval and schema normalization.",
            "Not a serum tumor-marker monitoring cohort.",
            "Not a patient-facing prediction validation source.",
        ],
    },
    {
        "id": "aacr_genie_bpc_brca",
        "name": "AACR GENIE BPC breast cancer",
        "source_url": "https://genie.cbioportal.org/api/studies/genie_bpc_brca?projection=SUMMARY",
        "secondary_source_url": "https://www.aacr.org/professionals/research/aacr-project-genie/",
        "access": "access-controlled candidate",
        "sample_count": None,
        "biomarker_fields": ["clinical-grade NGS", "ER", "PR", "HER2", "selected biomarkers"],
        "tumor_marker_fields": [],
        "outcomes": ["real-world treatment and survival outcomes where accessible"],
        "readiness": "access_controlled_future_candidate",
        "allowed_use": [
            "future real-world biomarker/treatment/outcome benchmark after access approval",
        ],
        "limitations": [
            "Not immediately available for local student training without access workflow.",
            "Do not claim validation from this source until data are actually mapped and evaluated.",
        ],
    },
    {
        "id": "nci_edrn_breast_reference_set",
        "name": "NCI-EDRN breast cancer reference set for circulating markers",
        "source_url": "https://edrn.nci.nih.gov/data-and-resources/publications/25471344-2344-construction-and-analysis-of-the-nci-edrn-breast-cancer-reference-set-for-circulating-markers-of-disease/",
        "secondary_source_url": "https://www.cancer.gov/about-cancer/diagnosis-staging/diagnosis/tumor-markers-list",
        "access": "research reference set / publication context",
        "sample_count": 832,
        "biomarker_fields": ["serum/plasma protein markers"],
        "tumor_marker_fields": ["CA15-3", "CEA-family/CEACAM5"],
        "outcomes": ["reference-set case/control labels, not treatment response"],
        "readiness": "tumor_marker_context_only_not_response_training",
        "allowed_use": [
            "tumor-marker limitation education",
            "serum-marker realism priors",
            "negative-control reminder against standalone tumor-marker prediction",
        ],
        "limitations": [
            "Not a longitudinal treatment-response cohort.",
            "Does not support standalone recurrence/progression prediction in NLCare.",
        ],
    },
    {
        "id": "nci_tumor_marker_common_use",
        "name": "NCI tumor-marker test reference",
        "source_url": "https://www.cancer.gov/about-cancer/diagnosis-staging/diagnosis/tumor-markers-list",
        "access": "public education/reference",
        "sample_count": None,
        "biomarker_fields": [],
        "tumor_marker_fields": ["CA15-3/CA27.29", "CEA"],
        "outcomes": [],
        "readiness": "context_only_patient_education",
        "allowed_use": [
            "patient-safe tumor-marker explanation",
            "RAG education source",
            "claim-boundary support",
        ],
        "limitations": [
            "Reference page, not a downloadable modeling cohort.",
            "Must not be used to train response predictors.",
        ],
    },
]


def build_public_biomarker_dataset_readiness(
    output_path: str | None = DEFAULT_OUTPUT_PATH,
    *,
    live_enrich: bool = False,
) -> dict[str, Any]:
    datasets = [dict(item) for item in DATASET_CANDIDATES]
    live_notes: list[dict[str, Any]] = []
    if live_enrich:
        live_notes = _enrich_with_public_apis(datasets)

    readiness_counts = _count_readiness(datasets)
    biomarker_benchmark = _load_json(BIOMARKER_BENCHMARK_PATH)
    full_ablation = _load_json(FULL_ABLATION_PATH)
    retraining_decision = _build_retraining_decision(biomarker_benchmark, full_ablation, readiness_counts)

    tumor_marker_train_ready = readiness_counts.get("tumor_marker_response_train_ready", 0)
    status = "strong" if readiness_counts["biomarker_external_candidate_count"] >= 4 and tumor_marker_train_ready == 0 else "acceptable"

    payload: dict[str, Any] = {
        "schema_version": "public_biomarker_dataset_readiness_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "summary": {
            **readiness_counts,
            "dataset_count": len(datasets),
            "best_immediate_external_benchmark": "breastdcedl",
            "best_schema_mapping_sources": ["metabric_cbioportal", "tcga_brca_pan_can_atlas"],
            "best_future_real_world_candidate": "aacr_genie_bpc_brca",
            "tumor_marker_policy": "context_only_until_longitudinal_treatment_response_data_exists",
        },
        "datasets": datasets,
        "live_api_enrichment": live_notes,
        "a_b_testing_plan": _build_ab_plan(),
        "retraining_decision": retraining_decision,
        "claim_boundary": (
            "This artifact is public-data readiness and engineering planning only. "
            "It is not clinical validation, does not prove clinical utility, does not validate "
            "biomarker or tumor-marker predictors, and does not justify treatment recommendations."
        ),
    }
    if output_path:
        path = ROOT_DIR / output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _count_readiness(datasets: list[dict[str, Any]]) -> dict[str, int]:
    biomarker_candidates = 0
    tumor_marker_context_only = 0
    tumor_marker_response_train_ready = 0
    access_controlled = 0
    mapped_or_api_ready = 0
    for dataset in datasets:
        readiness = str(dataset.get("readiness") or "")
        has_biomarkers = bool(dataset.get("biomarker_fields"))
        has_tumor_markers = bool(dataset.get("tumor_marker_fields"))
        if has_biomarkers and readiness not in {"context_only_patient_education"}:
            biomarker_candidates += 1
        if readiness in {"mapped_locally_external_benchmark", "api_schema_ready_biomarker_external_check"}:
            mapped_or_api_ready += 1
        if readiness.startswith("access_controlled"):
            access_controlled += 1
        if has_tumor_markers:
            if "response_training" in readiness or readiness.startswith("context_only"):
                tumor_marker_context_only += 1
            else:
                tumor_marker_response_train_ready += 1
    return {
        "biomarker_external_candidate_count": biomarker_candidates,
        "mapped_or_api_ready_count": mapped_or_api_ready,
        "access_controlled_candidate_count": access_controlled,
        "tumor_marker_context_only_count": tumor_marker_context_only,
        "tumor_marker_response_train_ready": tumor_marker_response_train_ready,
    }


def _build_retraining_decision(
    biomarker_benchmark: dict[str, Any],
    full_ablation: dict[str, Any],
    readiness_counts: dict[str, int],
) -> dict[str, Any]:
    biomarker_delta = _dig(biomarker_benchmark, ["deltas", "enhanced_vs_current_default_auroc_delta"])
    full_delta = _dig(full_ablation, ["deltas", "full_vs_clinical_auroc_delta"])
    full_recommended_use = _dig(full_ablation, ["recommendation", "recommended_use"]) or full_ablation.get("recommended_use")
    can_train_candidate = readiness_counts.get("mapped_or_api_ready_count", 0) >= 2
    production_retrain_now = False
    candidate_training_recommended = bool(can_train_candidate)
    return {
        "production_retrain_now": production_retrain_now,
        "candidate_training_recommended": candidate_training_recommended,
        "recommended_next_training": (
            "Train only an offline candidate table after mapping BreastDCEDL plus one cBioPortal source. "
            "Do not replace the champion until leakage, calibration, subgroup, counterfactual, and external checks pass."
        ),
        "why_not_promote_now": [
            "Current biomarker feature benchmark remains monitor_only.",
            "Tumor-marker sources found are context-only, not longitudinal response-training cohorts.",
            "No clinician-reviewed or real-world NLCare-style outcome labels are available.",
        ],
        "latest_synthetic_signals": {
            "enhanced_vs_current_default_auroc_delta": biomarker_delta,
            "full_vs_clinical_auroc_delta": full_delta,
            "full_feature_recommended_use": full_recommended_use,
        },
    }


def _build_ab_plan() -> dict[str, Any]:
    return {
        "baseline": "current synthetic champion with evidence-aware abstention",
        "candidate_a": "biomarker_context_candidate using ER/PR/HER2/Ki-67/genetic-readiness as contextual modifiers",
        "candidate_b": "imaging_plus_biomarker_candidate after BreastDCEDL mapping",
        "negative_control": "tumor_marker_standalone_candidate must be rejected",
        "promotion_requirements": [
            "No leakage audit failures.",
            "No worse safety/refusal behavior.",
            "Brier and ECE do not regress.",
            "Counterfactual unacceptable flips remain zero.",
            "External/public benchmark is reported separately from synthetic holdout metrics.",
            "Tumor markers never act as standalone recurrence/progression proof.",
        ],
    }


def _enrich_with_public_apis(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    by_id = {dataset["id"]: dataset for dataset in datasets}
    for dataset_id, url in {
        "metabric_cbioportal": "https://www.cbioportal.org/api/studies/brca_metabric?projection=SUMMARY",
        "tcga_brca_pan_can_atlas": "https://www.cbioportal.org/api/studies/brca_tcga_pan_can_atlas_2018?projection=SUMMARY",
    }.items():
        payload, error = _fetch_json(url)
        note: dict[str, Any] = {"dataset_id": dataset_id, "url": url, "ok": error is None}
        if payload:
            target = by_id.get(dataset_id)
            if target is not None:
                target["live_study_summary"] = {
                    key: payload.get(key)
                    for key in (
                        "name",
                        "description",
                        "citation",
                        "pmid",
                        "publicStudy",
                        "sequencedSampleCount",
                        "completeSampleCount",
                        "treatmentCount",
                    )
                    if key in payload
                }
            note["fields"] = sorted([key for key in payload if key.endswith("Count") or key in {"name", "publicStudy"}])
        else:
            note["error"] = error
        notes.append(note)
    payload, error = _fetch_json("https://api.gdc.cancer.gov/projects/TCGA-BRCA")
    notes.append({
        "dataset_id": "tcga_brca_gdc",
        "url": "https://api.gdc.cancer.gov/projects/TCGA-BRCA",
        "ok": error is None,
        "fields": sorted(payload.get("data", {}).keys()) if payload else [],
        "error": error,
    })
    return notes


def _fetch_json(url: str, timeout_seconds: int = 12) -> tuple[dict[str, Any] | None, str | None]:
    try:
        request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "NLCare-readiness/1.0"})
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - public APIs only
            return json.loads(response.read().decode("utf-8")), None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return None, str(exc)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _dig(payload: Any, path: list[Any]) -> Any:
    value = payload
    for key in path:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    return value
