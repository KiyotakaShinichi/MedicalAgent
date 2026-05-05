import json
import re
import sys
import tarfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
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
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_items = []
    for paper in PAPERS:
        result = download_paper(paper)
        manifest_items.append(result)
        print(f"{result['status']}: {result['pmcid']} -> {result.get('file_name') or result.get('reason')}")

    payload = {
        "schema_version": "research_paper_manifest_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "download_policy": "NCBI Open Access subset only. Do not add paywalled PDFs.",
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
            "path": str(file_path),
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
    return {
        "is_open_access": "<record " in xml_text and not error_match,
        "pdf": _ftp_to_https(pdf_match.group(1)) if pdf_match else None,
        "tgz": _ftp_to_https(tgz_match.group(1)) if tgz_match else None,
        "error": error_match.group(1) if error_match else None,
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
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="ignore")


def fetch_bytes(url):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=120) as response:
        return response.read()


def _ftp_to_https(url):
    if url.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
        return url.replace("ftp://ftp.ncbi.nlm.nih.gov/", "https://ftp.ncbi.nlm.nih.gov/")
    return url


def slugify(value):
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return value[:90]


if __name__ == "__main__":
    main()
