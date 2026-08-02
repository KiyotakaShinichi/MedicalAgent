import json
import re
import sys
import tarfile
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from xml.etree import ElementTree


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


OUTPUT_DIR = ROOT_DIR / "KnowledgeBase" / "raw" / "research_papers"
MANIFEST_PATH = OUTPUT_DIR / "research_papers_manifest.json"
USER_AGENT = "MedicalAgentPoC/1.0 (open-access research ingestion)"


PAPERS = [
    {
        "pmcid": "PMC3609956",
        "title": "The role of MRI in assessing residual disease and pCR after neoadjuvant chemotherapy",
        "landing_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3609956/",
        "topic": "mri_response_monitoring",
        "modality": ["MRI"],
        "stage": "neoadjuvant_treatment_response",
        "confidence": "peer_reviewed_open_access",
        "trust_level": "systematic_review",
    },
    {
        "pmcid": "PMC6702160",
        "title": "Early treatment response prediction using DCE-MRI tumor heterogeneity changes",
        "landing_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6702160/",
        "topic": "mri_response_prediction",
        "modality": ["DCE-MRI", "MRI"],
        "stage": "early_treatment_response",
        "confidence": "peer_reviewed_open_access",
        "trust_level": "research_paper",
    },
    {
        "pmcid": "PMC5500247",
        "title": "DCE-MRI texture features for early breast cancer therapy response prediction",
        "landing_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5500247/",
        "topic": "mri_radiomics_response",
        "modality": ["DCE-MRI", "MRI"],
        "stage": "early_treatment_response",
        "confidence": "peer_reviewed_open_access",
        "trust_level": "research_paper",
    },
    {
        "pmcid": "PMC10898895",
        "title": "Occurrence and risk factors of chemotherapy-induced neutropenia in breast cancer",
        "landing_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10898895/",
        "topic": "cbc_toxicity_monitoring",
        "modality": ["CBC", "clinical"],
        "stage": "chemotherapy_toxicity",
        "confidence": "peer_reviewed_open_access",
        "trust_level": "research_paper",
    },
    {
        "pmcid": "PMC6180804",
        "title": "Primary febrile neutropenia prophylaxis for FEC-D chemotherapy in breast cancer",
        "landing_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6180804/",
        "topic": "febrile_neutropenia_supportive_care",
        "modality": ["CBC", "clinical"],
        "stage": "chemotherapy_toxicity",
        "confidence": "peer_reviewed_open_access",
        "trust_level": "systematic_review",
    },
    {
        "pmcid": "PMC3987093",
        "title": "Patterns of chemotherapy-associated toxicity and supportive care in oncology practice",
        "landing_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3987093/",
        "topic": "chemotherapy_toxicity_supportive_care",
        "modality": ["CBC", "symptoms", "clinical"],
        "stage": "chemotherapy_toxicity",
        "confidence": "peer_reviewed_open_access",
        "trust_level": "research_paper",
    },
    {
        "pmcid": "PMC4998558",
        "title": "Prognostic value of chemotherapy-induced neutropenia at first cycle in invasive breast cancer",
        "landing_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4998558/",
        "topic": "cbc_toxicity_monitoring",
        "modality": ["CBC", "clinical"],
        "stage": "chemotherapy_toxicity",
        "confidence": "peer_reviewed_open_access",
        "trust_level": "research_paper",
    },
    {
        "pmcid": "PMC2846279",
        "title": "Prophylaxis of chemotherapy-induced febrile neutropenia with G-CSF",
        "landing_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC2846279/",
        "topic": "febrile_neutropenia_supportive_care",
        "modality": ["CBC", "clinical"],
        "stage": "chemotherapy_toxicity",
        "confidence": "peer_reviewed_open_access",
        "trust_level": "research_paper",
    },
    {
        "pmcid": "PMC3940691",
        "title": "Risk of hematological toxicities in a SWOG breast cancer chemotherapy trial",
        "landing_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3940691/",
        "topic": "cbc_toxicity_monitoring",
        "modality": ["CBC", "clinical"],
        "stage": "chemotherapy_toxicity",
        "confidence": "peer_reviewed_open_access",
        "trust_level": "research_paper",
    },
    {
        "pmcid": "PMC11400212",
        "title": "Essential components of an electronic patient-reported symptom monitoring and management system",
        "landing_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11400212/",
        "topic": "electronic_symptom_monitoring",
        "modality": ["symptoms", "patient-reported outcomes", "clinical workflow"],
        "stage": "active_treatment_monitoring",
        "confidence": "peer_reviewed_open_access_randomized_trial",
        "trust_level": "research_paper",
        "allowed_use": ["education", "monitoring_context"],
        "patient_facing_suitability": "education_with_boundary",
        "evidence_role": "workflow_design_evidence",
        "selection_rationale": "Randomized trial relevant to symptom-reporting and care-team follow-up workflow design.",
    },
    {
        "pmcid": "PMC4544753",
        "title": "ACMG and AMP standards and guidelines for the interpretation of sequence variants",
        "landing_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4544753/",
        "topic": "genetic_variant_interpretation_boundary",
        "modality": ["genetics", "laboratory interpretation"],
        "stage": "genetic_review",
        "confidence": "peer_reviewed_open_access_consensus_standard",
        "trust_level": "research_paper",
        "allowed_use": ["education", "monitoring_context"],
        "patient_facing_suitability": "review_routing_only",
        "evidence_role": "claim_boundary",
        "selection_rationale": "Defines professional variant-classification concepts used to keep automated VUS interpretation out of scope.",
    },
    {
        "pmcid": "PMC9158111",
        "title": "Variants of uncertain significance and reclassification in hereditary cancer testing",
        "landing_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9158111/",
        "topic": "vus_reclassification",
        "modality": ["genetics", "hereditary cancer"],
        "stage": "genetic_review",
        "confidence": "peer_reviewed_open_access",
        "trust_level": "research_paper",
        "allowed_use": ["education", "monitoring_context"],
        "patient_facing_suitability": "review_routing_only",
        "evidence_role": "claim_boundary",
        "selection_rationale": "Supports education about uncertainty and reclassification without treating a VUS as pathogenic.",
    },
    {
        "pmcid": "PMC7055088",
        "title": "Variants of uncertain significance in the era of high-throughput genome sequencing",
        "landing_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7055088/",
        "topic": "vus_breast_ovarian_cancer_context",
        "modality": ["genetics", "breast and ovarian cancer"],
        "stage": "genetic_review",
        "confidence": "peer_reviewed_open_access_review",
        "trust_level": "systematic_review",
        "allowed_use": ["education", "monitoring_context"],
        "patient_facing_suitability": "review_routing_only",
        "evidence_role": "uncertainty_context",
        "selection_rationale": "Breast and ovarian cancer context for why uncertain variants require qualified review.",
    },
    {
        "pmcid": "PMC2752999",
        "title": "Standards, options and recommendations for serum tumour markers in breast cancer",
        "landing_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC2752999/",
        "topic": "tumor_marker_limitations",
        "modality": ["tumor markers", "CA 15-3", "CEA"],
        "stage": "follow_up_context",
        "confidence": "peer_reviewed_open_access_guideline_summary",
        "trust_level": "research_paper",
        "allowed_use": ["education", "monitoring_context"],
        "patient_facing_suitability": "review_routing_only",
        "evidence_role": "claim_boundary",
        "selection_rationale": "Useful for explaining why serum markers are context and cannot independently establish diagnosis or recurrence.",
    },
    {
        "pmcid": "PMC10163167",
        "title": "Anxiety and depression in adult cancer patients: ESMO clinical practice guideline",
        "landing_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10163167/",
        "topic": "oncology_distress_review_routing",
        "modality": ["emotional distress", "patient-reported symptoms", "clinical workflow"],
        "stage": "supportive_care",
        "confidence": "peer_reviewed_open_access_guideline",
        "trust_level": "research_paper",
        "allowed_use": ["education", "monitoring_context"],
        "patient_facing_suitability": "review_routing_only",
        "evidence_role": "review_routing_context",
        "selection_rationale": "Supports cautious distress-screening and referral workflow concepts, not automated psychiatric diagnosis.",
    },
    {
        "pmcid": "PMC12452844",
        "title": "Use of PRO-CTCAE in oncology clinical trials",
        "landing_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12452844/",
        "topic": "patient_reported_outcome_measurement",
        "modality": ["symptoms", "PRO-CTCAE", "patient-reported outcomes"],
        "stage": "treatment_monitoring",
        "confidence": "peer_reviewed_open_access_review",
        "trust_level": "systematic_review",
        "allowed_use": ["education", "monitoring_context"],
        "patient_facing_suitability": "education_with_boundary",
        "evidence_role": "measurement_vocabulary",
        "selection_rationale": "Documents patient-reported symptom measurement vocabulary without turning self-report into clinician toxicity grading.",
    },
    {
        "pmcid": "PMC11211959",
        "title": "Remote monitoring app for endocrine therapy adherence among patients with early-stage breast cancer",
        "landing_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11211959/",
        "topic": "endocrine_monitoring_negative_result",
        "modality": ["symptoms", "medication adherence", "digital monitoring"],
        "stage": "endocrine_therapy_monitoring",
        "confidence": "peer_reviewed_open_access_randomized_trial",
        "trust_level": "research_paper",
        "allowed_use": ["education", "monitoring_context"],
        "patient_facing_suitability": "clinician_context_only",
        "evidence_role": "negative_result",
        "selection_rationale": "A useful negative result: the monitored intervention did not improve its primary adherence outcome.",
    },
    {
        "pmcid": "PMC6868829",
        "title": "Communication and handling of symptomatic adverse events during adjuvant breast cancer chemotherapy",
        "landing_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6868829/",
        "topic": "breast_cancer_epro_communication",
        "modality": ["symptoms", "patient-reported outcomes", "care-team communication"],
        "stage": "adjuvant_chemotherapy_monitoring",
        "confidence": "peer_reviewed_open_access",
        "trust_level": "research_paper",
        "allowed_use": ["education", "monitoring_context"],
        "patient_facing_suitability": "education_with_boundary",
        "evidence_role": "workflow_design_evidence",
        "selection_rationale": "Breast-cancer-specific evidence about communication around patient-reported adverse events.",
    },
    {
        "pmcid": "PMC12232703",
        "title": "Herb-drug interactions in oncology",
        "landing_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12232703/",
        "topic": "oncology_supplement_interaction_boundary",
        "modality": ["supplements", "medications", "pharmacist review"],
        "stage": "supportive_care",
        "confidence": "peer_reviewed_open_access_review",
        "trust_level": "systematic_review",
        "allowed_use": ["education", "monitoring_context"],
        "patient_facing_suitability": "review_routing_only",
        "evidence_role": "claim_boundary",
        "selection_rationale": "Supports routing supplement-interaction questions to oncology or pharmacy review without declaring products safe or unsafe.",
    },
    {
        "pmcid": "PMC6361332",
        "title": "Online tool for monitoring adverse events during cancer treatment: eRAPID field testing",
        "landing_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6361332/",
        "topic": "breast_cancer_remote_symptom_monitoring",
        "modality": ["symptoms", "patient-reported outcomes", "clinical workflow"],
        "stage": "active_treatment_monitoring",
        "confidence": "peer_reviewed_open_access_field_study",
        "trust_level": "research_paper",
        "allowed_use": ["education", "monitoring_context"],
        "patient_facing_suitability": "education_with_boundary",
        "evidence_role": "workflow_feasibility",
        "selection_rationale": "Breast-service field evidence for symptom capture and review workflow feasibility.",
    },
    {
        "pmcid": "PMC12053558",
        "title": "Web-based cancer symptom self-management system: a randomized clinical trial",
        "landing_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12053558/",
        "topic": "electronic_symptom_monitoring",
        "modality": ["symptoms", "patient-reported outcomes", "clinical workflow"],
        "stage": "treatment_and_survivorship_monitoring",
        "confidence": "peer_reviewed_open_access_randomized_trial",
        "trust_level": "research_paper",
        "allowed_use": ["education", "monitoring_context"],
        "patient_facing_suitability": "education_with_boundary",
        "evidence_role": "workflow_design_evidence",
        "selection_rationale": "Recent randomized evidence for EHR-integrated symptom reporting and self-management workflow design.",
    },
]

DEFAULT_NOT_ALLOWED_FOR = [
    "diagnosis",
    "treatment_selection_or_change",
    "dose_change",
    "prognosis_or_survival_estimate",
    "patient_specific_genetic_risk",
    "vus_reclassification",
    "tumor_marker_conclusion",
    "supplement_safety_clearance",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_items = []
    for paper in PAPERS:
        result = download_paper(paper)
        manifest_items.append(result)
        print(f"{result['status']}: {result['pmcid']} -> {result.get('file_name') or result.get('reason')}")

    payload = {
        "schema_version": "research_paper_manifest_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "download_policy": "NCBI Open Access subset only. Do not add paywalled PDFs.",
        "selection_policy": (
            "Internally selected engineering corpus for evidence retrieval, uncertainty, "
            "and review-routing tests. It is not a systematic review or clinical validation."
        ),
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "claim_boundary": (
            "Paper text may support source-backed education and evidence-routing tests only. "
            "It must not authorize diagnosis, treatment, dosage, prognosis, patient-specific "
            "genetic interpretation, tumor-marker conclusions, or supplement clearance."
        ),
        "items": manifest_items,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(OUTPUT_DIR),
        "manifest_path": str(MANIFEST_PATH),
        "downloaded": sum(1 for item in manifest_items if item["status"] in {"downloaded", "exists"}),
        "failed": sum(1 for item in manifest_items if item["status"] == "failed"),
    }, indent=2))


def download_paper(paper):
    try:
        links = discover_open_access_links(paper["pmcid"])
        identity = discover_bibliographic_identity(paper["pmcid"])
        paper = {
            **paper,
            **identity,
            "license": links.get("license") or "unknown",
            "oa_citation": links.get("citation"),
            "retracted": bool(links.get("retracted")),
            "allowed_use": paper.get("allowed_use") or ["education", "monitoring_context"],
            "patient_facing_suitability": paper.get("patient_facing_suitability") or "education_with_boundary",
            "evidence_role": paper.get("evidence_role") or "monitoring_context",
            "selection_rationale": paper.get("selection_rationale") or (
                "Retained from the original internally selected open-access monitoring corpus."
            ),
            "not_allowed_for": paper.get("not_allowed_for") or list(DEFAULT_NOT_ALLOWED_FOR),
        }
        if not links.get("is_open_access"):
            return {
                **paper,
                "status": "skipped",
                "reason": "No PDF or OA package found in NCBI Open Access subset.",
            }
        source_url = links.get("pdf") or links.get("tgz") or paper["landing_url"]
        file_name = f"{paper['pmcid']}_{slugify(paper['title'])}.txt"
        file_path = OUTPUT_DIR / file_name
        if file_path.exists() and file_path.stat().st_size > 1024:
            status = "exists"
            file_type = "open_access_full_text"
        else:
            text = ""
            if links.get("tgz"):
                try:
                    text = extract_text_from_oa_package(fetch_bytes(links["tgz"]))
                    file_type = "oa_package_text"
                except Exception:
                    text = ""
            if not text:
                text = extract_text_from_article_xml(fetch_bytes(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                    f"?db=pmc&id={re.sub(r'\D', '', paper['pmcid'])}&retmode=xml"
                    "&tool=NLCareEngineeringPrototype"
                ))
                file_type = "pmc_efetch_xml_text"
            if not text:
                text = extract_text_from_pmc_html(fetch_text(paper["landing_url"]))
                file_type = "pmc_html_text"
            if len(text) < 1000:
                raise ValueError("Open-access source did not contain extractable article text")
            file_path.write_text(text, encoding="utf-8")
            status = "downloaded"
        return {
            **paper,
            "status": status,
            "source_url": source_url,
            "file_type": file_type,
            "file_name": file_name,
            "path": file_path.relative_to(ROOT_DIR).as_posix(),
            "bytes": file_path.stat().st_size,
        }
    except Exception as exc:
        return {
            **paper,
            "status": "failed",
            "reason": str(exc),
        }


def discover_open_access_links(pmcid):
    xml_text = fetch_text(f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}")
    pdf_match = re.search(r'format="pdf"[^>]*href="([^"]+)"', xml_text)
    tgz_match = re.search(r'format="tgz"[^>]*href="([^"]+)"', xml_text)
    error_match = re.search(r"<error[^>]*>(.*?)</error>", xml_text)
    record_match = re.search(r"<record\s+([^>]+)>", xml_text)
    attributes = dict(re.findall(r'(\w+)="([^"]*)"', record_match.group(1))) if record_match else {}
    return {
        "is_open_access": "<record " in xml_text and not error_match,
        "pdf": _ftp_to_https(pdf_match.group(1)) if pdf_match else None,
        "tgz": _ftp_to_https(tgz_match.group(1)) if tgz_match else None,
        "license": attributes.get("license"),
        "citation": attributes.get("citation"),
        "retracted": attributes.get("retracted", "no").lower() in {"yes", "true", "1"},
        "error": error_match.group(1) if error_match else None,
    }


def discover_bibliographic_identity(pmcid):
    numeric_id = re.sub(r"\D", "", pmcid)
    idconv = json.loads(fetch_text(
        "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
        f"?ids={pmcid}&format=json&tool=NLCareEngineeringPrototype"
    ))
    record = (idconv.get("records") or [{}])[0]
    esummary = json.loads(fetch_text(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        f"?db=pmc&id={numeric_id}&retmode=json&tool=NLCareEngineeringPrototype"
    ))
    summary = (esummary.get("result") or {}).get(numeric_id) or {}
    return {
        "doi": record.get("doi"),
        "pmid": str(record.get("pmid")) if record.get("pmid") is not None else None,
        "publication_date": summary.get("pubdate") or summary.get("epubdate"),
        "journal": summary.get("source"),
        "publisher_title": summary.get("title"),
    }


def extract_text_from_oa_package(payload):
    with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
        nxml_members = [member for member in archive.getmembers() if member.name.endswith(".nxml")]
        if not nxml_members:
            return ""
        extracted = archive.extractfile(nxml_members[0])
        if extracted is None:
            return ""
        xml_payload = extracted.read()
    return extract_text_from_article_xml(xml_payload)


def extract_text_from_article_xml(xml_payload):
    root = ElementTree.fromstring(xml_payload)
    parts = []
    for element in root.iter():
        tag = element.tag.split("}")[-1]
        if tag in {"article-title", "title", "abstract", "sec", "p"}:
            text = " ".join(" ".join(element.itertext()).split())
            if text and (not parts or parts[-1] != text):
                parts.append(text)
    return "\n\n".join(parts)


def extract_text_from_pmc_html(html):
    html = re.sub(r"(?is)<(script|style|nav|footer|header).*?</\1>", " ", html)
    html = re.sub(r"(?i)</(h1|h2|h3|p|div|section|article|li)>", "\n\n", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    lines = [" ".join(line.split()) for line in text.splitlines()]
    lines = [line for line in lines if len(line) > 2]
    return "\n\n".join(lines)


def fetch_text(url):
    return fetch_bytes(url, timeout=60).decode("utf-8", errors="ignore")


def fetch_bytes(url, timeout=120):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code != 429 or attempt == 3:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
            time.sleep(max(1.0, delay))
    raise RuntimeError(f"Failed to fetch {url}")


def _ftp_to_https(url):
    if url.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
        return url.replace("ftp://ftp.ncbi.nlm.nih.gov/", "https://ftp.ncbi.nlm.nih.gov/")
    return url


def slugify(value):
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return value[:90]


if __name__ == "__main__":
    main()
