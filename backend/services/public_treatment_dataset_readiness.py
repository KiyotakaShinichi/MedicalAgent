from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_public_treatment_dataset_readiness.json"


TREATMENT_MODALITY_SCHEMA: list[dict[str, Any]] = [
    {
        "id": "surgery",
        "label": "Surgery",
        "examples": ["lumpectomy", "mastectomy", "sentinel lymph node biopsy", "axillary dissection"],
        "model_role": "timeline anchor / local-control context",
        "blocked_claims": ["recommend surgery type", "decide margins or operative plan"],
    },
    {
        "id": "radiation",
        "label": "Radiation therapy",
        "examples": ["whole-breast radiation", "regional nodal irradiation", "boost", "palliative radiation"],
        "model_role": "treatment-phase context and symptom/toxicity context",
        "blocked_claims": ["recommend radiation fields", "decide dose/fractions"],
    },
    {
        "id": "chemotherapy",
        "label": "Chemotherapy",
        "examples": ["AC-T", "dose-dense AC-T", "TC", "CMF", "carboplatin/paclitaxel"],
        "model_role": "cycle timing, CBC/toxicity context, response-monitoring context",
        "blocked_claims": ["start/stop chemo", "delay cycle", "change dose"],
    },
    {
        "id": "targeted_anti_her2",
        "label": "HER2-targeted therapy",
        "examples": ["trastuzumab", "pertuzumab", "T-DM1", "lapatinib", "neratinib"],
        "model_role": "biomarker-matched treatment context",
        "blocked_claims": ["choose HER2 regimen", "switch targeted therapy"],
    },
    {
        "id": "endocrine",
        "label": "Endocrine therapy",
        "examples": ["tamoxifen", "aromatase inhibitor", "ovarian suppression", "fulvestrant"],
        "model_role": "HR-positive treatment context and survivorship/adherence context",
        "blocked_claims": ["start/stop hormone therapy", "manage endocrine side effects without clinician"],
    },
    {
        "id": "immunotherapy",
        "label": "Immunotherapy",
        "examples": ["pembrolizumab"],
        "model_role": "TNBC regimen context and immune-toxicity review context",
        "blocked_claims": ["recommend immunotherapy", "interpret immune toxicity as diagnosis"],
    },
    {
        "id": "parp_inhibitor",
        "label": "PARP inhibitor",
        "examples": ["olaparib", "talazoparib"],
        "model_role": "germline/somatic context flag for clinician review",
        "blocked_claims": ["recommend PARP inhibitor from genetic result"],
    },
    {
        "id": "supportive_care",
        "label": "Supportive care medications",
        "examples": ["antiemetics", "growth factor", "pain control", "antibiotics"],
        "model_role": "symptom and toxicity-management context",
        "blocked_claims": ["prescribe supportive medication", "change dose"],
    },
]


TREATMENT_COMBINATION_PATTERNS: list[dict[str, Any]] = [
    {
        "id": "single_modality_endocrine",
        "modalities": ["endocrine"],
        "example_context": "HR-positive low-risk adjuvant context in some cases",
        "allowed_use": "structured timeline category only",
    },
    {
        "id": "chemo_only",
        "modalities": ["chemotherapy"],
        "example_context": "systemic chemotherapy backbone without targeted/endocrine flags in the record",
        "allowed_use": "cycle/lab monitoring context",
    },
    {
        "id": "chemo_plus_targeted",
        "modalities": ["chemotherapy", "targeted_anti_her2"],
        "example_context": "HER2-positive chemo plus HER2-targeted regimen context",
        "allowed_use": "biomarker-matched treatment context, not recommendation",
    },
    {
        "id": "chemo_plus_immunotherapy",
        "modalities": ["chemotherapy", "immunotherapy"],
        "example_context": "TNBC chemo-immunotherapy context",
        "allowed_use": "review-routing/toxicity context only",
    },
    {
        "id": "surgery_radiation_endocrine",
        "modalities": ["surgery", "radiation", "endocrine"],
        "example_context": "local therapy plus HR-positive systemic maintenance context",
        "allowed_use": "post-treatment/survivorship timeline category",
    },
    {
        "id": "chemo_surgery_radiation_endocrine",
        "modalities": ["chemotherapy", "surgery", "radiation", "endocrine"],
        "example_context": "multi-modality HR-positive treatment journey",
        "allowed_use": "timeline organization and missing-evidence context",
    },
    {
        "id": "chemo_targeted_surgery_radiation_endocrine",
        "modalities": ["chemotherapy", "targeted_anti_her2", "surgery", "radiation", "endocrine"],
        "example_context": "triple-positive/HER2-positive multi-modality journey",
        "allowed_use": "timeline organization and A/B feature-ablation candidate",
    },
]


PUBLIC_TREATMENT_DATASETS: list[dict[str, Any]] = [
    {
        "id": "aacr_genie_bpc_brca",
        "name": "AACR GENIE BPC Breast Cancer 1.0-public",
        "source_url": "https://www.aacr.org/professionals/research/aacr-project-genie/bpc/early-onset-brca/",
        "access": "public/access-controlled workflow depending on file/API route",
        "sample_count": 1130,
        "treatment_fields": [
            "cancer-directed drug regimens",
            "HER2-directed treatment histories",
            "real-world treatment response",
            "PFS imaging",
            "PFS medical oncology",
            "overall survival",
        ],
        "modality_coverage": ["chemotherapy", "targeted_anti_her2", "endocrine", "genomic testing"],
        "combination_support": "best future real-world treatment-combination benchmark",
        "readiness": "future_high_value_treatment_outcomes_candidate",
        "limitations": [
            "Requires access/terms workflow before local training.",
            "Not yet mapped into NLCare's longitudinal CBC/symptom/imaging timeline.",
        ],
    },
    {
        "id": "seer_breast",
        "name": "NCI SEER breast cancer registry variables",
        "source_url": "https://seer.cancer.gov/data/seerstat/",
        "access": "SEER data-use agreement / SEER*Stat",
        "sample_count": None,
        "treatment_fields": [
            "surgery",
            "radiation therapy first course",
            "chemotherapy fields with documented limitations",
            "breast subtype",
            "ER/PR/HER2",
            "response neoadjuvant therapy in modern schemas",
        ],
        "modality_coverage": ["surgery", "radiation", "chemotherapy", "biomarker subtype"],
        "combination_support": "coarse first-course treatment combination and population distribution checks",
        "readiness": "agreement_required_population_treatment_distribution_candidate",
        "limitations": [
            "Registry treatment variables are coarse and can be incomplete.",
            "Not a serial CBC/symptom/treatment-cycle response dataset.",
        ],
    },
    {
        "id": "seer_medicare",
        "name": "SEER-Medicare linked claims",
        "source_url": "https://healthcaredelivery.cancer.gov/seermedicare/aboutdata/program.html",
        "access": "application/DUA required",
        "sample_count": None,
        "treatment_fields": [
            "claims-derived surgery",
            "radiation",
            "chemotherapy",
            "hormonal therapy",
            "HER2-targeted agents",
        ],
        "modality_coverage": ["surgery", "radiation", "chemotherapy", "endocrine", "targeted_anti_her2"],
        "combination_support": "strong claims-based treatment combination source after access approval",
        "readiness": "restricted_future_claims_benchmark",
        "limitations": [
            "Requires application, DUA, and claims expertise.",
            "Medicare population skews older; subgroup boundaries matter.",
        ],
    },
    {
        "id": "breastdcedl_ispy2",
        "name": "BreastDCEDL I-SPY2 subset",
        "source_url": "https://zenodo.org/records/17578255",
        "access": "public non-commercial research dataset",
        "sample_count": 982,
        "treatment_fields": ["neoadjuvant trial context", "pCR", "HR", "HER2", "MammaPrint risk"],
        "modality_coverage": ["chemotherapy_trial_context", "imaging_response", "biomarker subtype"],
        "combination_support": "excellent imaging-response benchmark but not full real-world combination timeline",
        "readiness": "mapped_external_response_candidate",
        "limitations": [
            "Does not provide NLCare-style medication-by-cycle, radiation, surgery, endocrine, and supportive-care timeline.",
            "Use for pCR/imaging response, not treatment-choice recommendation.",
        ],
    },
    {
        "id": "duke_breast_mri_tcia",
        "name": "Duke Breast Cancer MRI / TCIA",
        "source_url": "https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/",
        "access": "public TCIA collection",
        "sample_count": 922,
        "treatment_fields": ["neoadjuvant chemotherapy context", "pathology/receptor metadata"],
        "modality_coverage": ["imaging_response", "chemotherapy_context", "ER/PR/HER2"],
        "combination_support": "imaging plus receptor/treatment-context external check",
        "readiness": "manual_mapping_imaging_treatment_context_candidate",
        "limitations": [
            "Not a complete multimodal treatment-combination timeline.",
            "Requires image/data download and normalization.",
        ],
    },
    {
        "id": "tcga_brca_gdc",
        "name": "TCGA-BRCA via GDC",
        "source_url": "https://api.gdc.cancer.gov/projects/TCGA-BRCA",
        "access": "public API/open files; controlled data require token",
        "sample_count": 1098,
        "treatment_fields": ["diagnoses.treatments treatment_type", "treatment_or_therapy"],
        "modality_coverage": ["coarse treatment metadata", "omics", "survival"],
        "combination_support": "coarse treatment metadata only",
        "readiness": "api_schema_coarse_treatment_check",
        "limitations": [
            "Treatment fields are not rich enough for detailed regimen-combination training.",
            "Best used for schema sanity checks, not clinical response validation.",
        ],
    },
    {
        "id": "clinicaltrials_gov_breast_protocols",
        "name": "ClinicalTrials.gov breast cancer protocols",
        "source_url": "https://clinicaltrials.gov/",
        "access": "public registry",
        "sample_count": None,
        "treatment_fields": ["intervention names", "arms", "eligibility", "outcome definitions"],
        "modality_coverage": ["regimen vocabulary", "trial-arm combinations"],
        "combination_support": "regimen vocabulary source, not patient-level outcome training",
        "readiness": "vocabulary_only_not_patient_training",
        "limitations": [
            "Protocol registry, not patient-level timeline data.",
            "Useful for controlled treatment vocabulary and KB enrichment only.",
        ],
    },
]


def build_public_treatment_dataset_readiness(
    output_path: str | None = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    datasets = [dict(item) for item in PUBLIC_TREATMENT_DATASETS]
    readiness_counts = _count_readiness(datasets)
    payload: dict[str, Any] = {
        "schema_version": "public_treatment_dataset_readiness_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if readiness_counts["treatment_combination_candidate_count"] >= 5 else "acceptable",
        "summary": {
            **readiness_counts,
            "dataset_count": len(datasets),
            "best_future_real_world_treatment_dataset": "aacr_genie_bpc_brca",
            "best_population_distribution_source": "seer_breast",
            "best_immediate_public_response_context": "breastdcedl_ispy2",
            "immediate_full_treatment_combo_training_ready": 0,
        },
        "treatment_modality_schema": TREATMENT_MODALITY_SCHEMA,
        "treatment_combination_patterns": TREATMENT_COMBINATION_PATTERNS,
        "datasets": datasets,
        "controlled_feature_plan": {
            "synthetic_now": [
                "treatment modality flags",
                "combination pattern",
                "sequence phase: neoadjuvant/adjuvant/metastatic/survivorship",
                "multi-modality count",
                "dose delay/reduction/interruption flags",
            ],
            "external_now": [
                "BreastDCEDL/I-SPY2 pCR plus HR/HER2 imaging-response benchmark",
                "TCGA/GDC coarse treatment metadata schema check",
            ],
            "future_after_access": [
                "GENIE BPC treatment histories and real-world outcomes",
                "SEER/SEER-Medicare population treatment combinations",
            ],
        },
        "claim_boundary": (
            "This artifact maps treatment-combination data sources and a controlled treatment vocabulary. "
            "It does not recommend any treatment, compare treatment efficacy for real patients, or validate "
            "NLCare as a clinical treatment-response predictor."
        ),
    }
    if output_path:
        path = ROOT_DIR / output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _count_readiness(datasets: list[dict[str, Any]]) -> dict[str, int]:
    combination_candidates = 0
    access_required = 0
    immediate_public_response_context = 0
    vocabulary_only = 0
    for dataset in datasets:
        readiness = str(dataset.get("readiness") or "")
        if "candidate" in readiness or "check" in readiness:
            combination_candidates += 1
        if any(token in readiness for token in ("future", "agreement_required", "restricted")):
            access_required += 1
        if readiness in {"mapped_external_response_candidate", "manual_mapping_imaging_treatment_context_candidate", "api_schema_coarse_treatment_check"}:
            immediate_public_response_context += 1
        if readiness.startswith("vocabulary_only"):
            vocabulary_only += 1
    return {
        "treatment_combination_candidate_count": combination_candidates,
        "access_or_agreement_required_count": access_required,
        "immediate_public_response_context_count": immediate_public_response_context,
        "vocabulary_only_count": vocabulary_only,
    }


def load_public_treatment_dataset_readiness(path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    file_path = ROOT_DIR / path
    if file_path.exists():
        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return build_public_treatment_dataset_readiness(output_path=path)
