"""Built-in knowledge-base seed corpus for the patient agent.

This module is the canonical home for ``KNOWLEDGE_SNIPPETS`` — the
~18 hand-curated reference snippets that ``agent_rag`` merges with the
ingested chunks at runtime.  Snippets here are the project's
"always-on" knowledge contract: fever-during-chemo wording, pCR
definition, CBC monitoring boundaries, supplement-safety rules,
imaging report monitoring policy, portal help text, etc.

Each entry follows the same shape as a chunk produced by the KB
ingestion pipeline so the merger in :func:`agent_rag._knowledge_snippets`
treats them uniformly.

Why this lives in its own module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The data literal is ~290 lines.  Keeping it inline in ``agent_rag.py``
inflated the orchestrator file by 30% with content that is pure data,
not logic.  Moving it makes ``agent_rag`` navigable for someone reading
the pipeline rather than the seed corpus.

Adding / editing snippets
~~~~~~~~~~~~~~~~~~~~~~~~~
Edit ``KNOWLEDGE_SNIPPETS`` below and re-run any RAG eval that exercises
the affected text.  After a change, call
``backend.services.agent_rag._invalidate_kb_cache()`` (or restart the
process) so the merged corpus reloads.
"""
from __future__ import annotations


KNOWLEDGE_SNIPPETS: list[dict] = [
    {
        "id": "cdc-fever-chemo",
        "parent_id": "infection-safety",
        "title": "Fever during chemotherapy",
        "source_name": "CDC",
        "source_url": "https://www.cdc.gov/cancer-preventing-infections/patients/fever.html",
        "tags": ["fever", "infection", "chemotherapy", "urgent", "wbc", "neutropenia"],
        "builtin": True,
        "text": (
            "During chemotherapy, fever can be a sign of infection risk and should be treated as urgent. "
            "A patient should contact the oncology team immediately for fever or feeling seriously unwell."
        ),
    },
    {
        "id": "nci-side-effects",
        "parent_id": "treatment-side-effects",
        "title": "Treatment side effects",
        "source_name": "National Cancer Institute",
        "source_url": "https://www.cancer.gov/about-cancer/treatment/side-effects",
        "tags": ["side effects", "symptoms", "fatigue", "nausea", "doctor", "treatment"],
        "builtin": True,
        "text": (
            "Cancer treatment can cause side effects, and patients should tell their doctor about symptoms so the care team "
            "can help manage problems. Monitoring symptoms over time is useful for clinical review."
        ),
    },
    {
        "id": "nci-breast-chemo",
        "parent_id": "breast-treatment-basics",
        "title": "Chemotherapy for breast cancer",
        "source_name": "National Cancer Institute",
        "source_url": "https://www.cancer.gov/types/breast/treatment/chemotherapy",
        "tags": ["breast cancer", "chemotherapy", "neoadjuvant", "adjuvant", "treatment"],
        "builtin": True,
        "text": (
            "Breast cancer chemotherapy may be given before surgery (neoadjuvant) to shrink tumor burden or after surgery to reduce recurrence risk. "
            "The exact plan depends on clinician-directed staging, subtype, and treatment goals."
        ),
    },
    {
        "id": "acs-chemo-side-effects",
        "parent_id": "treatment-side-effects",
        "title": "Chemotherapy side effects",
        "source_name": "American Cancer Society",
        "source_url": "https://www.cancer.org/cancer/managing-cancer/treatment-types/chemotherapy/chemotherapy-side-effects.html",
        "tags": ["chemotherapy", "wbc", "hemoglobin", "platelets", "cbc", "infection", "anemia", "fatigue"],
        "builtin": True,
        "text": (
            "Chemotherapy side effects can include lower white blood cells, anemia, fatigue, nausea, and infection risk. "
            "CBC trends help clinicians monitor toxicity and recovery during treatment."
        ),
    },
    {
        "id": "project-pcr-definition",
        "parent_id": "response-modeling",
        "title": "pCR in the project",
        "source_name": "Project model card",
        "source_url": "MODEL_CARD.md",
        "tags": ["pcr", "pathologic complete response", "response", "mri", "classification", "score"],
        "builtin": True,
        "text": (
            "In this PoC, pCR means pathologic complete response - defined as the absence of residual invasive tumor after neoadjuvant treatment. "
            "It is used as a classification target in breast cancer research datasets. "
            "The project treats it as an engineering label, not as a diagnosis or patient-facing clinical conclusion."
        ),
    },
    {
        "id": "project-monitoring-score",
        "parent_id": "response-modeling",
        "title": "Monitoring score boundary",
        "source_name": "Project safety policy",
        "source_url": "README.md",
        "tags": ["score", "probability", "model", "response", "monitoring", "classification"],
        "builtin": True,
        "text": (
            "The treatment monitoring score is an exploratory engineering signal that combines model response signals with CBC and symptom concerns. "
            "It is for trend discussion and clinician review, not a treatment decision."
        ),
    },
    {
        "id": "nci-her2-breast",
        "parent_id": "breast-treatment-basics",
        "title": "HER2 in breast cancer",
        "source_name": "National Cancer Institute",
        "source_url": "https://www.cancer.gov/types/breast/treatment/her2",
        "tags": ["her2", "breast", "cancer", "targeted therapy", "receptor"],
        "builtin": True,
        "text": (
            "HER2 is a protein receptor that can be overexpressed in some breast cancers. "
            "HER2-positive breast cancer status is determined by testing and affects treatment planning."
        ),
    },
    {
        "id": "nci-chemo-nadir",
        "parent_id": "treatment-side-effects",
        "title": "Chemotherapy nadir and blood counts",
        "source_name": "National Cancer Institute",
        "source_url": "https://www.cancer.gov/about-cancer/treatment/side-effects/low-blood-counts",
        "tags": ["nadir", "chemotherapy", "wbc", "cbc", "neutropenia", "blood counts"],
        "builtin": True,
        "text": (
            "A nadir is the lowest point in blood cell counts after a chemotherapy dose. "
            "The nadir typically occurs 7 to 14 days after chemotherapy and increases infection risk."
        ),
    },
    {
        "id": "nci-febrile-neutropenia",
        "parent_id": "treatment-side-effects",
        "title": "Febrile neutropenia during chemotherapy",
        "source_name": "National Cancer Institute",
        "source_url": "https://www.cancer.gov/about-cancer/treatment/side-effects/infection/infection-hp-pdq",
        "tags": ["neutropenia", "fever", "chemotherapy", "infection", "urgent", "anc"],
        "builtin": True,
        "text": (
            "Febrile neutropenia is a fever occurring when neutrophil counts are critically low during chemotherapy. "
            "Neutropenia with fever requires urgent oncology evaluation due to high infection risk."
        ),
    },
    {
        "id": "nci-chemo-dose-delay",
        "parent_id": "treatment-side-effects",
        "title": "Chemotherapy dose delays",
        "source_name": "National Cancer Institute",
        "source_url": "https://www.cancer.gov/about-cancer/treatment/drugs",
        "tags": ["dose", "delay", "chemotherapy", "blood counts", "toxicity", "treatment"],
        "builtin": True,
        "text": (
            "Chemotherapy dose delays occur when blood counts are too low to safely proceed. "
            "A clinician evaluates whether to delay the next dose based on CBC results and recovery."
        ),
    },
    {
        "id": "nci-msk-supplement-safety",
        "parent_id": "supportive-care-safety",
        "title": "Supplements during cancer treatment",
        "source_name": "NCI / NCCIH / MSK",
        "source_url": "https://www.cancer.gov/about-cancer/treatment/cam/patient/dietary-interactions-pdq",
        "tags": [
            "supplement",
            "supplements",
            "antioxidant",
            "turmeric",
            "herbal",
            "vitamin",
            "chemotherapy",
            "interactions",
            "oncology",
            "pharmacist",
        ],
        "builtin": True,
        "text": (
            "Supplements, antioxidant products, vitamins, herbs, and turmeric can interact with chemotherapy, radiation, "
            "targeted therapy, surgery, or supportive medicines. Patients should tell the oncology care team or oncology "
            "pharmacist about every supplement they use or are considering. This system can provide general education and "
            "log supplement questions for review, but it does not recommend starting, stopping, replacing, or dosing a "
            "supplement as cancer treatment."
        ),
    },
    {
        "id": "curated-triple-negative-basics",
        "parent_id": "breast-treatment-basics",
        "title": "Triple-negative breast cancer",
        "source_name": "Curated NCI breast cancer education",
        "source_url": "https://www.cancer.gov/types/breast/hp/breast-treatment-pdq",
        "tags": ["triple-negative", "tnbc", "er", "pr", "her2", "subtype", "breast cancer"],
        "builtin": True,
        "text": (
            "Triple-negative breast cancer means the tumor is ER negative, PR negative, and HER2 negative by clinical testing. "
            "It is a breast cancer subtype used by clinicians for treatment planning. OncoTrack can explain the term, but it cannot classify a patient from chat text."
        ),
    },
    {
        "id": "curated-stage-iv-basics",
        "parent_id": "breast-treatment-basics",
        "title": "Stage IV breast cancer boundary",
        "source_name": "Curated NCI breast cancer education",
        "source_url": "https://www.cancer.gov/types/breast/hp/breast-treatment-pdq",
        "tags": ["stage iv", "metastatic", "staging", "breast cancer", "clinician"],
        "builtin": True,
        "text": (
            "Stage IV breast cancer generally means metastatic disease, or cancer that has spread to distant organs. "
            "Staging requires clinician interpretation of pathology and imaging. The assistant must not assign a patient's stage."
        ),
    },
    {
        "id": "curated-taxane-neuropathy",
        "parent_id": "treatment-side-effects",
        "title": "Paclitaxel and neuropathy monitoring",
        "source_name": "Curated breast cancer treatment education",
        "source_url": "https://www.cancer.org/cancer/types/breast-cancer/treatment/chemotherapy-for-breast-cancer.html",
        "tags": ["paclitaxel", "docetaxel", "taxane", "neuropathy", "tingling", "numbness"],
        "builtin": True,
        "text": (
            "Paclitaxel and other taxane chemotherapy drugs can be associated with neuropathy, such as tingling, numbness, burning, or pain in hands or feet. "
            "Patients can log neuropathy severity for review, but the assistant must not recommend dose changes."
        ),
    },
    {
        "id": "curated-platelets-bleeding",
        "parent_id": "cbc-monitoring",
        "title": "Platelets and bleeding risk",
        "source_name": "Curated CBC monitoring education",
        "source_url": "https://www.cancer.gov/about-cancer/treatment/side-effects/low-blood-counts",
        "tags": ["platelets", "cbc", "bleeding", "clotting", "chemotherapy"],
        "builtin": True,
        "text": (
            "Platelets help blood clot. Low platelet counts during treatment can increase bruising or bleeding risk. "
            "Bleeding symptoms should be logged and reviewed by the oncology care team."
        ),
    },
    {
        "id": "curated-acupuncture-supportive-care",
        "parent_id": "integrative-supportive-care",
        "title": "Acupuncture and acupressure supportive care boundary",
        "source_name": "Curated ASCO/SIO integrative oncology education",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/29889605/",
        "tags": ["acupuncture", "acupressure", "nausea", "supportive care", "oncology"],
        "builtin": True,
        "text": (
            "Acupuncture or acupressure may be discussed as supportive care for symptoms such as nausea in some oncology settings. "
            "Patients should ask the oncology team before using it, especially with low platelets, infection risk, anticoagulants, lymphedema risk, wounds, or implanted devices."
        ),
    },
    {
        "id": "curated-st-johns-wort-safety",
        "parent_id": "supplement-safety",
        "title": "St. Johns wort interaction safety",
        "source_name": "Curated supplement interaction safety",
        "source_url": "https://www.mskcc.org/cancer-care/patient-education/herbal-remedies-and-treatment",
        "tags": ["st johns wort", "supplement", "herbal", "interact", "interaction", "oncology", "pharmacist"],
        "builtin": True,
        "text": (
            "St. Johns wort can interact with many medicines through drug metabolism pathways. "
            "During cancer treatment, patients should not start St. Johns wort without review by the oncology care team or oncology pharmacist."
        ),
    },
    {
        "id": "curated-ct-ascites-monitoring",
        "parent_id": "imaging-monitoring",
        "title": "CT ascites report wording monitoring",
        "source_name": "Curated imaging report monitoring",
        "source_url": "KnowledgeBase/raw/curated_medical_kb/05_imaging/ct_ascites_report_monitoring.md",
        "tags": [
            "ct",
            "ascites",
            "peritoneal",
            "imaging",
            "clinician",
            "oncology monitoring",
            "report wording",
            "metastasis",
        ],
        "topic": "ct_report_monitoring",
        "modality": ["CT", "imaging", "ascites"],
        "care_stage": "treatment_monitoring",
        "trust_level": "patient_education",
        "builtin": True,
        "text": (
            "CT reports may mention ascites, peritoneal nodularity, liver lesions, effusion, or other findings. "
            "OncoTrack can track the exact report wording for clinician review alongside symptoms, labs, prior imaging, "
            "and treatment history. It must not diagnose metastasis, recurrence, or treatment response from CT wording alone."
        ),
    },
    {
        "id": "curated-model-signal-boundary",
        "parent_id": "portal-help",
        "title": "Model signal explanation",
        "source_name": "OncoTrack project documentation",
        "source_url": "README.md",
        "tags": ["model signal", "monitoring score", "portal", "exploratory", "not diagnosis", "clinician review"],
        "builtin": True,
        "text": (
            "The OncoTrack model signal is an exploratory engineering signal in this proof of concept. "
            "It is not a diagnosis, not a treatment recommendation, and not clinical validation. It helps organize clinician review."
        ),
    },
    {
        "id": "portal-upload-guide",
        "parent_id": "portal-help",
        "title": "What patients can upload",
        "source_name": "Project patient portal guide",
        "source_url": "README.md",
        "tags": ["upload", "portal", "cbc", "mri", "symptoms", "medications", "labs"],
        "builtin": True,
        "text": (
            "The patient portal is designed to store CBC/lab values, MRI or imaging files, imaging report text, medications, treatments, "
            "and symptoms so changes can be summarized over time."
        ),
    },
]


__all__ = ["KNOWLEDGE_SNIPPETS"]
