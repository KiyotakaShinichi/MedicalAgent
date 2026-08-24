"""Configuration and source aliases for the frozen RAG baseline evaluator."""

from __future__ import annotations

from typing import Any


CONFIGURATIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "bm25_only",
        "label": "BM25 only",
        "description": "Sparse BM25 lexical retrieval over the frozen KB corpus; no query rewriting.",
    },
    {
        "id": "faiss_dense_only",
        "label": "FAISS dense only",
        "description": "Dense/vector score ordering from the active local index; falls back honestly if dense FAISS is unavailable.",
    },
    {
        "id": "hybrid_rrf",
        "label": "Dense + sparse hybrid RRF",
        "description": "Active dense/sparse hybrid retrieval with reciprocal-rank-style fusion; no query rewriting.",
    },
    {
        "id": "hybrid_rrf_query_rewrite",
        "label": "Hybrid + query rewrite",
        "description": "Hybrid retrieval using the agent query rewriting/decomposition output.",
    },
    {
        "id": "hybrid_rrf_query_rewrite_parent_child",
        "label": "Hybrid + rewrite + parent-child",
        "description": "Hybrid retrieval with query rewriting plus parent-child context expansion.",
    },
    {
        "id": "hybrid_rrf_query_rewrite_parent_child_source_tier",
        "label": "Hybrid + rewrite + parent-child + source tiers",
        "description": "Full compared retrieval stack with source-tier/allowed-use filtering before context selection.",
    },
    {
        "id": "hybrid_rrf_query_rewrite_parent_child_source_tier_pruned",
        "label": "Full stack + citation-context pruner",
        "experimental": True,
        "positioning": "negative_result_not_promoted",
        "description": (
            "Full stack with the citation_context_pruner applied between source-tier "
            "filtering and citation assembly.  Eval-path experiment only — not wired "
            "into the live patient agent."
        ),
    },
)

REFUSAL_INTENTS = {
    "urgent_escalation",
    "genetic_counselor_review",
    "tumor_marker_boundary",
    "pharmacist_or_clinician_review",
    "treatment_refusal",
    "prognosis_refusal",
    "diagnosis_refusal",
    "privacy_refusal",
}

# Source normalization keeps logical gold labels comparable to the current KB
# source IDs. It is not a ranking tweak and does not inspect retrieved text.
LOGICAL_SOURCE_ALIASES: dict[str, set[str]] = {
    "nci-her2-breast": {
        "nci-her2-breast",
        "her2 in breast cancer",
        "breast-treatment-basics",
        "national cancer institute",
    },
    "curated-her2-basics": {
        "curated-her2-basics",
        "nci-her2-breast",
        "her2 in breast cancer",
        "breast-treatment-basics",
    },
    "cbc-monitoring": {
        "cbc-monitoring",
        "curated-wbc-neutropenia",
        "cbc labs and trend monitoring",
        "cbc, anc, hemoglobin, and platelet monitoring reference",
        "0185db088c803c80",
        "36b7a3ffdb9205a4",
        "927cf11805df9019d710",
        "f6726c194bf1f479171f",
    },
    "curated-wbc-neutropenia": {
        "curated-wbc-neutropenia",
        "cbc-monitoring",
        "side effects and red flags during breast cancer treatment",
        "treatment-side-effects",
        "3ca1dfefbd3147b0",
        "c30ab0b49f328562e76f",
        "nci-febrile-neutropenia",
        "febrile neutropenia during chemotherapy",
    },
    "infection-safety": {
        "infection-safety",
        "cdc",
        "cdc-fever-chemo",
        "fever during chemotherapy",
        "nci-febrile-neutropenia",
        "febrile neutropenia during chemotherapy",
        "treatment-side-effects",
        # Discovered by content match (see latest_source_alias_coverage.json).
        "9a6347c207d53299",  # Hematology, Bleeding, and Infection Review Reference
    },
    "curated-fever-neutropenia": {
        "curated-fever-neutropenia",
        "nci-febrile-neutropenia",
        "febrile neutropenia during chemotherapy",
        "infection-safety",
        "treatment-side-effects",
    },
    "imaging-monitoring": {
        "imaging-monitoring",
        "curated-mri-response-terms",
        "imaging report monitoring: mri, ct, ultrasound, and response language",
        "mri, ct, ultrasound, and imaging response terms reference",
        "a734a844daed9ef7",
        "33ef73acba84d60bd7a1",
        "87ec22bc66c88b40ea76",
        "7cd1e3e1103a156a",
    },
    "curated-mri-response-terms": {
        "curated-mri-response-terms",
        "imaging-monitoring",
        "imaging report monitoring: mri, ct, ultrasound, and response language",
        "mri, ct, ultrasound, and imaging response terms reference",
        "a734a844daed9ef7",
        "33ef73acba84d60bd7a1",
        "87ec22bc66c88b40ea76",
        "7cd1e3e1103a156a",
        # Discovered by content match (see latest_source_alias_coverage.json).
        "2524619e8115a75d",  # DCE-MRI texture features for early breast cancer therapy response prediction
        "2a9f2ed73f0b189c",  # Early treatment response prediction using DCE-MRI tumor heterogeneity
    },
    "genetic-counseling": {
        "genetic-counseling",
        "curated-vus-boundary",
        "genetic counseling readiness and family history intake",
        "germline testing, somatic testing, vus, and multigene panels",
        "genetics, biomarker, and tumor marker safety terms reference",
        "22d463a5a12d490af4c6",
        "29f0f5dda9789b7e",
        "4787d2a42440789f",
        "eafe5c100c4cd819b6fa",
        "917264d81e3123c0d2a8",
        # Discovered by content match against KB titles
        # (see latest_source_alias_coverage.json, 2026-05-27 diagnostic).
        "664fb49bb1343408",  # Family History Readiness Depth Reference
        "ef3bcc511aad3c2c",  # Genetic Counseling Readiness and Family History Intake
    },
    "curated-vus-boundary": {
        "curated-vus-boundary",
        "genetic-counseling",
        "vus",
        "germline testing, somatic testing, vus, and multigene panels",
        "genetics, biomarker, and tumor marker safety terms reference",
        "29f0f5dda9789b7e",
        "4787d2a42440789f",
        "eafe5c100c4cd819b6fa",
        "917264d81e3123c0d2a8",
    },
    "tumor-marker-context": {
        "tumor-marker-context",
        "curated-tumor-marker-limitations",
        "minimum evidence and medical claim boundaries",
        "genetics, biomarker, and tumor marker safety terms reference",
        "28cfcee61ce1e4a4",
        "4787d2a42440789f",
        "972b1b8be879098562a7",
        "150bf2854b59cec640b1",
        "917264d81e3123c0d2a8",
        # Discovered by content match (see latest_source_alias_coverage.json).
        "5598e2371d2713c4",  # Breast Cancer Biomarkers and Tumor Marker Safety
    },
    "curated-tumor-marker-limitations": {
        "curated-tumor-marker-limitations",
        "tumor-marker-context",
        "minimum evidence and medical claim boundaries",
        "genetics, biomarker, and tumor marker safety terms reference",
        "28cfcee61ce1e4a4",
        "4787d2a42440789f",
        "972b1b8be879098562a7",
        "150bf2854b59cec640b1",
        "917264d81e3123c0d2a8",
        # Discovered by content match (see latest_source_alias_coverage.json).
        "5598e2371d2713c4",  # Breast Cancer Biomarkers and Tumor Marker Safety
    },
    "supplement-safety": {
        "supplement-safety",
        "curated-st-johns-wort",
        "curated-st-johns-wort-safety",
        "nci-msk-supplement-safety",
        "supplements during cancer treatment",
        "curated supplement interaction safety",
        "supplement and natural product safety by product reference",
        "6649c1bba1cd7799",
        "2c9cf580eb45af0e",
        "bd077c510af8e9bb2107",
        # Discovered by content match (see latest_source_alias_coverage.json).
        "918edc260afd2d63",  # Diagnosis, Treatment, and Supplement Safety Boundaries
    },
    "curated-st-johns-wort": {
        "curated-st-johns-wort",
        "curated-st-johns-wort-safety",
        "supplement-safety",
        "st. johns wort interaction safety",
        "st johns wort interaction safety",
    },
    "project safety policy": {
        "project safety policy",
        "project-monitoring-score",
        "monitoring score boundary",
        "diagnosis, treatment, and supplement safety boundaries",
        "minimum evidence and medical claim boundaries",
        "response-modeling",
        "918edc260afd2d63",
        "28cfcee61ce1e4a4",
        "b4b9ee5dfff5d9bb4a84",
    },
    "treatment-side-effects": {
        "treatment-side-effects",
        "acs-chemo-side-effects",
        "side effects and red flags during breast cancer treatment",
        "3ca1dfefbd3147b0",
        "1d8b472e73bcd9696d15",
        # Discovered by content match (see latest_source_alias_coverage.json).
        "24de6c8ad0379f43",  # GI Symptoms, Mouth Sores, Neuropathy, and Fatigue Reference
        "d50090fd5d38a39d",  # Symptom Red Flags and Review Hints During Treatment
    },
    "portal-help": {
        "portal-help",
        "portal-help-upload",
        "portal-help-symptom-entry",
        "portal-help-lab-results",
        "portal-help-mri-upload",
        "patient portal help",
        "using the patient portal tools",
        # Discovered by content match (see latest_source_alias_coverage.json).
        "c35c9264029ff9c9",  # NLCare Portal Help and Data Entry
        "479e2ce02e7d9e05",  # Patient Portal Workflow Reference
    },
}
