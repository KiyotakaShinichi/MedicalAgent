"""Static vocabulary and regex policy floors for agent safety routing."""

from __future__ import annotations

import re


# ─── Vocabulary tables ───────────────────────────────────────────────────────
#
# Kept as module-level frozen sets so a future caller can introspect what
# the safety contract actually catches without re-running the agent.

DECISION_TERMS: tuple[str, ...] = (
    "should i stop",
    "should i start",
    "should i change",
    "should i delay",
    "can i stop",
    "can i start",
    "can i change",
    "can i decrease",
    "can i increase",
    "write me a prescription",
    "prescription for",
    "prescribe me",
    "can i delay",
    "do i need to delay",
    # Note: "what dose" alone is too greedy — it false-positives on
    # "what dose-dense AC means".  Specific-phrasing matches keep the
    # treatment-decision intent without flagging the educational form.
    "what dose of",
    "what dose should",
    "what dose do i",
    "dose reduction",
    "usual dose reduction",
    "change my dose",
    "what treatment should i take",
    "should i change chemo",
    "should i change chemotherapy",
    "should i change treatment",
    "should i change surgery",
    "increase my dose",
    "decrease my dose",
    "increase my",
    "decrease my",
    "stop chemo",
    "stop chemotherapy",
    "delay chemo",
    "delay chemotherapy",
    "delay treatment",
    "delay my next chemo",
    "skip chemo",
    "skip treatment",
    "instead of chemotherapy",
    "instead of chemo",
    "replace chemotherapy",
    "replace chemo",
    "replace my chemotherapy",
    "replace my chemo",
    "use turmeric instead",
    "use supplements instead",
    "herbal instead of chemo",
    "supplement instead of chemo",
    "supplements instead of chemo",
    "supplement instead of chemotherapy",
    "supplements instead of chemotherapy",
    "dapat ko bang itigil",
    "pwede ko bang itigil",
    "pwede ko bang ihinto",
    "itigil chemo",
    "ihinto chemo",
    "itigil chemotherapy",
    "itigil yung chemo",
    "itigil ang chemo",
    "palitan dose",
    "taasan dose",
    "babaan dose",
    "baguhin dosage",
    "baguhin gamot",
    "anong dose",
    "anong dosage",
    "mag reseta",
    "ireseta",
    "debo parar",
    "parar la quimioterapia",
    "que dosis",
    "qué dosis",
    "dosis debo cambiar",
    "停止化疗",
    "改變什麼劑量",
    "改变什么剂量",
    "прекратить химиотерапию",
    "какую дозу",
    "изменить дозу",
)

DIAGNOSTIC_TERMS: tuple[str, ...] = (
    "do i have cancer",
    "do i have breast cancer",
    "do i have brca",
    "do i have brca1",
    "do i have brca2",
    "is this lump cancer",
    "is this cancer",
    "tell me if this is cancer",
    "tell me whether i have breast cancer",
    "based on my mri",
    "lesion in my mri malignant",
    "is the lesion",
    "malignant",
    "biopsy result means cancer",
    "result means cancer",
    "vus means positive",
    "vus mean positive",
    "vus positive",
    "vus in brca mean positive",
    "vus the same as positive",
    "variant of uncertain significance mean cancer",
    "vus means i have the mutation",
    "vus am i positive",
    "positive for cancer risk",
    "vus sa brca positive",
    "brca vus means positive",
    "positive na ba",
    "genetic mutation that means cancer",
    "brca1 mean i will definitely get cancer",
    "will i get cancer",
    "will my children get cancer",
    "will my relatives get cancer",
    "will my family get cancer",
    "confirm that i have hereditary breast cancer",
    "chek2 mean i have cancer",
    "germline result means cancer",
    "hereditary breast cancer",
    "is it metastatic",
    "has my cancer spread",
    "cancer spread to my bones",
    "do i have metastatic",
    "do i have metastatic disease",
    "do i have metastasis",
    "am i cancer free",
    "is my cancer gone",
    "does that mean my cancer came back",
    "does this mean my cancer came back",
    "does that mean recurrence",
    "does this mean recurrence",
    "confirm that my recurrence is back",
    "recurrence is back",
    "recurrence na ba",
    "ca 15-3 mataas",
    "ca 15-3 result proof",
    "rising cea proof",
    "proof of metastasis",
    "ca 27 29 is elevated",
    "ca 27.29 is elevated",
    "tumor marker high",
    "cancer is back",
    "diagnose me",
    "tell me if i have cancer",
    "tell me whether i have cancer",
    "will i survive",
    "will i beat",
    "how long do i have",
    "my prognosis",
    "survival rate",
    "survival chances",
    "may cancer ba ako",
    "metastatic ba",
    "meron na ba akong metastatic",
    "may metastatic ba",
    "kumalat na ba",
    "bumalik na ba",
    "ibig sabihin ba bumalik",
    "gumaling na ba ako",
    "cancer free na ba ako",
    "diagnose mo ako",
    "gaano katagal buhay ko",
    "mabubuhay ba ako",
    "ilang buwan",
    "buwan na lang",
    "prognosis ko",
    "confirma recurrencia",
    "alto confirma recurrencia",
    "cuanto me queda",
    "复发",
    "证明复发",
    "сколько мне осталось жить",
)

URGENT_TERMS: tuple[str, ...] = (
    "fever",
    "chest pain",
    "cannot breathe",
    "shortness of breath",
    "uncontrolled bleeding",
    "heavy bleeding",
    "bleeding now",
    "actively bleeding",
    "cannot stop bleeding",
    "bloody discharge",
    "bloody breast discharge",
    "blood discharge",
    "blood breast discharge",
    "suicidal",
    "self harm",
    "lagnat",
    "nilalagnat",
)

RESEARCH_AUTHORITY_OVERCLAIM_PATTERN = re.compile(
    r"\b(papers?|research|stud(?:y|ies)|trials?|literature|evidence)\b"
    r".{0,100}\b(prove|proves|proved|confirm|confirms|validate|validates|guarantee|establish)\b"
    r".{0,100}\b(this|our|the)\s+(app|prototype|agent|assistant|system|platform|portal)\b"
    r"|\b(this|our|the)\s+(app|prototype|agent|assistant|system|platform|portal)\b"
    r".{0,100}\b(prove|proves|proved|confirm|confirms|validate|validates|guarantee|establish)\b"
    r".{0,100}\b(safe|clinical|real patients?|patient care|benefit|ready)\b",
    re.IGNORECASE,
)

PERSONAL_SUPPLEMENT_SAFETY_PATTERN = re.compile(
    r"\b(safe|safely|okay|ok|pwede|puwede|ligtas)\b.{0,55}"
    r"\b(herb|herbal|supplement|vitamin|turmeric|cannabis|cbd|natural product)\b"
    r"|\b(herb|herbal|supplement|vitamin|turmeric|cannabis|cbd|natural product)\b"
    r".{0,55}\b(safe|safely|okay|ok|pwede|puwede|ligtas)\b",
    re.IGNORECASE,
)

ACTIVE_TREATMENT_CUE_PATTERN = re.compile(
    r"\b(during|while|habang|with|kasabay|on)\b.{0,35}"
    r"\b(chemo|chemotherapy|radiation|treatment|therapy|tamoxifen|trastuzumab|paclitaxel)\b",
    re.IGNORECASE,
)

_CROSS_PERSON_DATA_REQUEST_PATTERN = re.compile(
    r"\b(reveal|show|give|provide|export|open|retrieve|fetch|disclose)\b"
    r".{0,100}\b(contact details?|phone|email|address|record|chart|labs?|data|account)\b"
    r".{0,100}\b(other|another|previous|prior|next|person before me|person after me)\b"
    r"|\b(other|another|previous|prior|next|person before me|person after me)\b"
    r".{0,100}\b(person|patient|portal user|account holder)\b"
    r".{0,100}\b(contact details?|phone|email|address|record|chart|labs?|data|account)\b"
    r"|\bperson\b.{0,80}\b(?:used|accessed|visited)\b.{0,40}\bportal\b"
    r".{0,50}\b(?:before|after)\s+me\b",
    re.IGNORECASE,
)

_PRIVATE_RECORD_DISCLOSURE_PATTERN = re.compile(
    r"\b(?:disclose|reveal|show|send|give|provide|export|copy|read|recite|"
    r"retrieve|fetch|open|access)\b.{0,90}"
    r"\b(?:private|confidential|patient|person(?:'s)?)\b.{0,55}"
    r"\b(?:contact details?|phone|email|address|record|chart|labs?|data|account)\b"
    r"|\b(?:private|confidential|patient|person(?:'s)?)\b.{0,55}"
    r"\b(?:contact details?|phone|email|address|record|chart|labs?|data|account)\b"
    r".{0,90}\b(?:disclose|reveal|show|send|give|provide|export|copy|read|"
    r"recite|retrieve|fetch|open|access)\b",
    re.IGNORECASE,
)

_TUMOR_MARKER_CONCLUSION_PATTERN = re.compile(
    r"\b(?:tumou?r markers?|markers?|cea|ca\s*15-?3|ca\s*27\.?29)\b"
    r".{0,100}\b(?:conclude|assume|settle|prove|confirm|establish|means?)\b"
    r".{0,80}\b(?:progress(?:ed|ion)?|recurrence|recurred|metastasis|"
    r"metastatic|spread|treatment failed|therapy failed)\b"
    r"|\b(?:conclude|assume|settle|prove|confirm|establish)\b.{0,80}"
    r"\b(?:progress(?:ed|ion)?|recurrence|recurred|metastasis|metastatic|spread)\b"
    r".{0,100}\b(?:tumou?r markers?|markers?|cea|ca\s*15-?3|ca\s*27\.?29)\b",
    re.IGNORECASE,
)

# Tagalog/Taglish "do I have <disease>?" requests.
#
# The English diagnostic-request rule below keys on the surface form
# "do/could/might I have", which Tagalog cannot produce: possession is
# existential (`meron/mayroon/may ... ako`) and the question is marked by the
# particle `ba`. So "Base sa symptoms ko, meron na ba akong metastatic?" — the
# exact translation of a request the English rule stops — matched nothing and
# was routed as `general_support` / `education_or_tracking`, i.e. a diagnosis
# request from a Taglish speaker crossed the safety boundary that the same
# request in English does not.
#
# Two orders are accepted because Tagalog permits both:
#   particle-first  "meron na ba akong metastatic"
#   particle-last   "kanser na ba ako", "may bukol ba ako"
#
# The interrogative particle `ba` and a first-person/subject marker are both
# required, so purely educational phrasings such as "Ano ba yung metastatic?"
# ("what is metastatic?") stay educational: they carry no possession claim
# about the speaker. Over-triggering here would make the assistant refuse
# ordinary Tagalog health education, which is its own harm.
_TAGLISH_DIAGNOSTIC_REQUEST_PATTERN = re.compile(
    r"(?:\b(?:meron|mayroon|may)\b(?:\s+na)?\s+\bba\b.{0,60}?"
    r"\b(?:cancer|kanser|recurrence|metasta\w*|bukol|tumor|brca|mutation)\b"
    r"|\b(?:meron|mayroon|may)\b.{0,40}?"
    r"\b(?:cancer|kanser|recurrence|metasta\w*|bukol|tumor)\b\s*"
    r"(?:na\s+)?\bba\s+(?:ako|ito|ang)\b"
    r"|\b(?:cancer|kanser|recurrence|metasta\w*|bukol|tumor)\b\s*"
    r"(?:na\s+)?\bba\s+(?:ako|ito)\b)",
    re.IGNORECASE,
)

_MULTILINGUAL_TUMOR_MARKER_CONCLUSION_PATTERN = re.compile(
    r"(?:(?<![a-z0-9])(?:tumou?r markers?|markers?|cea|"
    r"ca(?:\s*is\s*e|is\s*e|\s*2t\s*29|2t\s*29)|"
    r"marcadores? tumorales?)(?![a-z0-9])|(?:肿瘤标志物|癌胚抗原))"
    r".{0,100}(?:(?:\b(?:confirma?|demuestra|prueba|significa)\b)|"
    r"(?:证明|证实|表明|意味着))"
    r".{0,80}(?:(?:\b(?:recurrenci[ae]|progresi[oó]n|met[aá]stasis)\b)|"
    r"(?:复发|进展|转移))",
    re.IGNORECASE,
)
