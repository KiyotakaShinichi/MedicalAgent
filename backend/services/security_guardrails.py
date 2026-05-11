import base64
import binascii
import re
import unicodedata

from backend.services.local_llm import assess_security_with_local_llm


LEET_TRANSLATION = str.maketrans({
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
    "!": "i",
})


PROMPT_CONTROL_PHRASES = [
    "ignore previous",
    "ignore all previous",
    "disregard instructions",
    "forget your rules",
    "override system",
    "bypass safety",
    "bypass guardrail",
    "disable safety",
    "jailbreak",
    "developer mode",
    "dan mode",
    "act as my doctor",
    "act as admin",
    "pretend you are admin",
    "system prompt",
    "developer message",
    "hidden instruction",
    "secret instruction",
    "reveal prompt",
    "print prompt",
    "show prompt",
    "chain of thought",
    "internal policy",
    "tool output",
    "raw tool",
    "run sql",
    "execute sql",
]


TAGALOG_ATTACK_PHRASES = [
    "wag sundin",
    "huwag sundin",
    "wag mong sundin",
    "huwag mong sundin",
    "kalimutan mo ang instructions",
    "kalimutan mo yung instructions",
    "balewalain mo ang instructions",
    "balewalain mo yung rules",
    "ipakita ang system prompt",
    "ipakita yung system prompt",
    "pakita mo ang prompt",
    "pakita mo yung prompt",
    "kunin ang database",
    "kunin yung database",
    "ibigay ang database",
    "ibigay yung database",
    "ilabas ang database",
    "ilabas yung database",
    "lahat ng pasyente",
    "ibang pasyente",
    "record ng pasyente",
    "records ng pasyente",
    "medical record ng iba",
    "datos ng pasyente",
    "tagong instruction",
    "sikretong instruction",
]


MULTILINGUAL_ATTACK_PHRASES = [
    "ignora las instrucciones",
    "ignora instrucciones",
    "muestra la base de datos",
    "dame la base de datos",
    "registros de pacientes",
    "otro paciente",
    "ignore les instructions",
    "ignorez les instructions",
    "affichez la base",
    "montre la base de donnees",
    "dossiers des patients",
    "ignori le istruzioni",
    "mostra il database",
    "ignorar instrucoes",
    "mostre o banco de dados",
    "ignoriere alle anweisungen",
    "zeige die datenbank",
    "patientendaten",
    "игнорируй инструкции",
    "покажи базу данных",
    "данные пациентов",
    "忽略指令",
    "忽略以前的指令",
    "显示系统提示",
    "显示数据库",
    "患者数据",
    "患者記録",
    "システムプロンプトを表示",
    "データベースを表示",
    "지시를 무시",
    "데이터베이스 보여",
    "환자 기록",
    "تجاهل التعليمات",
    "اعرض قاعدة البيانات",
    "سجلات المرضى",
    "निर्देशों को अनदेखा",
    "डेटाबेस दिखाओ",
    "मरीजों का डेटा",
]


SENSITIVE_TARGET_TERMS = [
    "database",
    "db",
    "sql",
    "sqlite",
    "schema",
    "tables",
    "patient records",
    "patient info",
    "patient data",
    "all patients",
    "other patient",
    "another patient",
    "someone else",
    "medical record",
    "phi",
    "personal health information",
    "knowledge base",
    "kb",
    "source documents",
    "raw documents",
    "dataset",
    "api key",
    "env",
    ".env",
    "secret",
    "token",
    "credentials",
    "pasyente",
    "ibang pasyente",
    "lahat ng patient",
    "lahat ng pasyente",
    "datos",
    "record",
    "registros de pacientes",
    "patientendaten",
    "данные пациентов",
    "患者数据",
    "患者記録",
    "환자 기록",
    "سجلات المرضى",
    "मरीजों का डेटा",
]


EXFILTRATION_VERBS = [
    "dump",
    "export",
    "download",
    "extract",
    "exfiltrate",
    "leak",
    "steal",
    "show",
    "list",
    "print",
    "reveal",
    "give me",
    "send",
    "copy",
    "query",
    "select",
    "kunin",
    "ibigay",
    "ipakita",
    "pakita",
    "ilabas",
    "kopyahin",
    "muestra",
    "dame",
    "donne",
    "mostra",
    "mostre",
    "zeige",
    "anzeigen",
    "покажи",
    "раскрой",
    "显示",
    "展示",
    "見せて",
    "表示",
    "보여",
    "اعرض",
    "اظهر",
    "दिखाओ",
]


URGENT_MEDICAL_PHRASES = [
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "difficulty breathing",
    "shortness of breath",
    "severe bleeding",
    "fainting",
    "confused",
    "fever",
    "suicidal",
    "self harm",
    "kill myself",
    "matinding lagnat",
    "hindi makahinga",
    "sakit sa dibdib",
    "magpakamatay",
    "no puedo respirar",
    "dolor en el pecho",
    "fiebre",
    "suicida",
    "je ne peux pas respirer",
    "douleur thoracique",
    "fievre",
    "kann nicht atmen",
    "brustschmerzen",
    "лихорадка",
    "не могу дышать",
    "боль в груди",
    "胸痛",
    "呼吸できない",
    "発熱",
    "发烧",
    "无法呼吸",
    "열",
    "숨을 못 쉬",
    "가슴 통증",
    "حمى",
    "ألم في الصدر",
    "لا أستطيع التنفس",
    "बुखार",
    "सांस नहीं",
    "सीने में दर्द",
]


SQL_OR_FILE_PATTERNS = [
    r"\bselect\s+\*\s+from\b",
    r"\bdrop\s+table\b",
    r"\binsert\s+into\b",
    r"\bpragma\s+table_info\b",
    r"\bunion\s+select\b",
    r"\bopen\s*\(\s*[\"']?medical_agent\.db",
    r"\bcat\s+\.env\b",
    r"\btype\s+\.env\b",
    r"\bget-content\s+\.env\b",
]


def detect_prompt_injection_or_exfiltration(text):
    normalized = normalize_security_text(text)
    variants = [normalized] + [normalize_security_text(item) for item in _decoded_variants(text)]
    issues = []
    signals = []

    phrase_groups = [
        ("prompt_injection_or_jailbreak", PROMPT_CONTROL_PHRASES),
        ("prompt_injection_or_jailbreak", TAGALOG_ATTACK_PHRASES),
        ("prompt_injection_or_jailbreak", MULTILINGUAL_ATTACK_PHRASES),
    ]
    for category, phrases in phrase_groups:
        matches = []
        for variant in variants:
            matches.extend(_phrase_matches(variant, variant.replace(" ", ""), phrases))
        if matches:
            issues.append(category)
            signals.extend({"category": category, "match": match} for match in sorted(set(matches))[:5])

    sql_matches = []
    for pattern in SQL_OR_FILE_PATTERNS:
        if any(re.search(pattern, variant) for variant in variants):
            sql_matches.append(pattern)
    if sql_matches:
        issues.append("database_or_file_access_attempt")
        signals.extend({"category": "database_or_file_access_attempt", "match": match} for match in sql_matches[:5])

    if any(_has_exfiltration_intent(variant) for variant in variants):
        issues.append("sensitive_data_exfiltration_attempt")
        signals.append({"category": "sensitive_data_exfiltration_attempt", "match": "verb+protected_target"})

    if any(_asks_for_other_patient(variant) for variant in variants):
        issues.append("privacy_boundary_request")
        signals.append({"category": "privacy_boundary_request", "match": "other/all patient data"})

    medical = detect_multilingual_medical_danger(text)
    if medical["detected"]:
        issues.append("urgent_medical_or_self_harm")
        signals.extend({"category": "urgent_medical_or_self_harm", "match": item} for item in medical["matches"][:5])

    benign_self_entry = not issues and (
        _is_benign_self_data_entry(normalized)
        or _is_benign_self_memory_query(normalized)
    )
    obvious_low_risk = not issues and _is_obvious_low_risk_support_or_education(normalized)
    should_ask_llm = bool(issues) or not obvious_low_risk
    llm_assessment = (
        assess_security_with_local_llm(
            text,
            deterministic_context={"issues": sorted(set(issues)), "signals": signals[:10]},
        )
        if should_ask_llm
        else {
            "available": False,
            "reason": "skipped_for_obvious_low_risk_support_or_education",
        }
    )
    llm_wants_block = (
        llm_assessment.get("available")
        and llm_assessment.get("blocked")
        and float(llm_assessment.get("confidence") or 0) >= 0.7
    )
    llm_confidence = float(llm_assessment.get("confidence") or 0)
    llm_can_block_without_deterministic_issue = (
        _has_security_or_privacy_anchor(normalized)
        or llm_confidence >= 0.95
    )
    if (
        llm_wants_block
        and not benign_self_entry
        and not obvious_low_risk
        and (issues or llm_can_block_without_deterministic_issue)
    ):
        issues.extend(str(issue) for issue in llm_assessment.get("issues") or ["llm_security_boundary"])
        signals.append({
            "category": "local_llm_security_assessment",
            "match": llm_assessment.get("reason") or "blocked",
            "confidence": llm_assessment.get("confidence"),
        })
    elif llm_wants_block and (benign_self_entry or obvious_low_risk):
        signals.append({
            "category": "llm_security_assessment_suppressed",
            "match": "benign self-scoped or low-risk support/education wording",
            "confidence": llm_assessment.get("confidence"),
        })

    issues = sorted(set(issues))
    blocked = bool(issues)
    return {
        "blocked": blocked,
        "status": "failed" if blocked else "passed",
        "issues": issues,
        "signals": signals[:10],
        "confidence": _confidence(issues, signals),
        "llm_assessment": llm_assessment,
        "medical_danger": medical,
        "normalized_text": normalized[:500],
        "message": (
            "Security guardrail blocked suspected prompt-injection, jailbreak, privacy-boundary, data-exfiltration, or urgent danger intent."
            if blocked else "Security guardrail passed."
        ),
    }


def normalize_security_text(text):
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = "".join(char for char in value if unicodedata.category(char) not in {"Cf", "Cc"})
    value = _strip_diacritics(value).lower().translate(LEET_TRANSLATION)
    value = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", value)
    value = re.sub(r"[_\-./\\|*~`'\"()\[\]{}:;,+?<>]+", " ", value)
    value = re.sub(r"(.)\1{3,}", r"\1\1", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def detect_multilingual_medical_danger(text):
    normalized = normalize_security_text(text)
    compact = normalized.replace(" ", "")
    matches = []
    for phrase in URGENT_MEDICAL_PHRASES:
        clean = normalize_security_text(phrase)
        if clean and (clean in normalized or clean.replace(" ", "") in compact):
            matches.append(phrase)
    return {
        "detected": bool(matches),
        "matches": sorted(set(matches))[:10],
    }


def _phrase_matches(normalized, compact, phrases):
    matches = []
    for phrase in phrases:
        clean = normalize_security_text(phrase)
        if clean and (clean in normalized or clean.replace(" ", "") in compact):
            matches.append(phrase)
    return matches


def _has_exfiltration_intent(normalized):
    has_verb = any(_term_present(normalized, verb) for verb in EXFILTRATION_VERBS)
    has_target = any(_term_present(normalized, target) for target in SENSITIVE_TARGET_TERMS)
    return has_verb and has_target


def _asks_for_other_patient(normalized):
    patient_scope_terms = [
        "all patients",
        "other patient",
        "another patient",
        "someone else",
        "results for patient",
        "data for patient",
        "records for patient",
        "ibang pasyente",
        "lahat ng pasyente",
        "lahat ng patient",
        "records ng pasyente",
        "patient records",
        "registros de pacientes",
        "patientendaten",
        "данные пациентов",
        "患者数据",
        "患者記録",
        "환자 기록",
        "سجلات المرضى",
        "मरीजों का डेटा",
    ]
    return any(_term_present(normalized, term) for term in patient_scope_terms)


def _is_benign_self_data_entry(normalized):
    action_terms = [
        "where can i put",
        "where do i put",
        "how do i upload",
        "how can i upload",
        "upload my",
        "put my",
        "save my",
        "log my",
        "record my",
        "report my",
        "add my",
        "enter my",
    ]
    self_data_terms = [
        "my cbc",
        "my lab",
        "my labs",
        "my medication",
        "my medications",
        "my symptom",
        "my symptoms",
        "my mri",
        "my scan",
        "my upload",
    ]
    portal_terms = ["portal", "upload", "uploads", "cbc", "medication", "symptoms", "mri"]
    return (
        any(term in normalized for term in action_terms)
        and any(term in normalized for term in self_data_terms)
        and any(term in normalized for term in portal_terms)
    )


def _is_benign_self_memory_query(normalized):
    memory_terms = [
        "what did i tell you",
        "what did i say",
        "what was my last message",
        "what did i mention",
        "remember what i said",
        "remember what i told you",
        "my chat history",
    ]
    risky_scope_terms = [
        "other patient",
        "another patient",
        "all patients",
        "database",
        "system prompt",
        "developer message",
        "secret",
        "api key",
    ]
    return any(term in normalized for term in memory_terms) and not any(term in normalized for term in risky_scope_terms)


def _is_obvious_low_risk_support_or_education(normalized):
    if _has_security_or_privacy_anchor(normalized):
        return False
    if len(normalized.split()) <= 6 and any(term in normalized for term in [
        "hi",
        "hello",
        "hey",
        "kumusta",
        "kamusta",
        "who are you",
        "how are you",
        "what can you do",
    ]):
        return True
    low_risk_terms = [
        "what is pcr",
        "what does pcr",
        "cbc trends",
        "what is cbc",
        "what do cbc",
        "wbc",
        "hemoglobin",
        "platelets",
        "side effect",
        "breast cancer monitoring",
        "chemotherapy",
    ]
    return any(term in normalized for term in low_risk_terms)


def _has_security_or_privacy_anchor(normalized):
    anchor_terms = [
        "ignore",
        "disregard",
        "bypass",
        "override",
        "jailbreak",
        "developer mode",
        "system prompt",
        "developer message",
        "hidden instruction",
        "chain of thought",
        "database",
        "sql",
        "schema",
        "table",
        "other patient",
        "all patients",
        "patient records",
        "patient data",
        "knowledge base",
        "raw documents",
        "api key",
        "secret",
        "token",
        "credentials",
        "wag sundin",
        "huwag sundin",
        "pasyente",
        "ibang pasyente",
        "lahat ng pasyente",
        "base de datos",
        "registros de pacientes",
    ]
    return any(_term_present(normalized, term) for term in anchor_terms)


def _term_present(normalized, term):
    clean = normalize_security_text(term)
    if not clean:
        return False
    if " " in clean:
        return clean in normalized
    if _has_word_boundary_script(clean):
        return re.search(rf"\b{re.escape(clean)}\b", normalized) is not None
    return clean in normalized


def _strip_diacritics(value):
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _decoded_variants(text):
    raw = str(text or "")
    decoded = []
    for candidate in re.findall(r"(?<![A-Za-z0-9+/=])([A-Za-z0-9+/]{16,}={0,2})(?![A-Za-z0-9+/=])", raw)[:3]:
        try:
            padded = candidate + ("=" * ((4 - len(candidate) % 4) % 4))
            value = base64.b64decode(padded, validate=True)
            decoded_text = value.decode("utf-8")
        except (binascii.Error, ValueError, UnicodeDecodeError):
            continue
        if decoded_text and len(decoded_text) <= 2000:
            decoded.append(decoded_text)
    for candidate in re.findall(r"\b(?:[0-9a-fA-F]{2}){8,}\b", raw)[:3]:
        try:
            decoded_text = bytes.fromhex(candidate).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
        if decoded_text and len(decoded_text) <= 2000:
            decoded.append(decoded_text)
    return decoded


def _has_word_boundary_script(value):
    return all(ord(char) < 128 for char in value)


def _confidence(issues, signals):
    if not issues:
        return 0.0
    score = 0.55 + min(0.4, 0.1 * len(issues)) + min(0.05, 0.01 * len(signals))
    return round(min(score, 0.99), 3)
