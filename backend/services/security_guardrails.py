import re
import unicodedata


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
    "montre la base de donnees",
    "dossiers des patients",
    "ignori le istruzioni",
    "mostra il database",
    "ignorar instrucoes",
    "mostre o banco de dados",
    "患者データ",
    "系统提示",
    "忽略指令",
    "显示数据库",
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
    compact = normalized.replace(" ", "")
    issues = []
    signals = []

    phrase_groups = [
        ("prompt_injection_or_jailbreak", PROMPT_CONTROL_PHRASES),
        ("prompt_injection_or_jailbreak", TAGALOG_ATTACK_PHRASES),
        ("prompt_injection_or_jailbreak", MULTILINGUAL_ATTACK_PHRASES),
    ]
    for category, phrases in phrase_groups:
        matches = _phrase_matches(normalized, compact, phrases)
        if matches:
            issues.append(category)
            signals.extend({"category": category, "match": match} for match in matches[:5])

    sql_matches = []
    for pattern in SQL_OR_FILE_PATTERNS:
        if re.search(pattern, normalized):
            sql_matches.append(pattern)
    if sql_matches:
        issues.append("database_or_file_access_attempt")
        signals.extend({"category": "database_or_file_access_attempt", "match": match} for match in sql_matches[:5])

    if _has_exfiltration_intent(normalized):
        issues.append("sensitive_data_exfiltration_attempt")
        signals.append({"category": "sensitive_data_exfiltration_attempt", "match": "verb+protected_target"})

    if _asks_for_other_patient(normalized):
        issues.append("privacy_boundary_request")
        signals.append({"category": "privacy_boundary_request", "match": "other/all patient data"})

    issues = sorted(set(issues))
    blocked = bool(issues)
    return {
        "blocked": blocked,
        "status": "failed" if blocked else "passed",
        "issues": issues,
        "signals": signals[:10],
        "confidence": _confidence(issues, signals),
        "normalized_text": normalized[:500],
        "message": (
            "Security guardrail blocked suspected prompt-injection, jailbreak, privacy-boundary, or data-exfiltration intent."
            if blocked else "Security guardrail passed."
        ),
    }


def normalize_security_text(text):
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = "".join(char for char in value if unicodedata.category(char) not in {"Cf", "Cc"})
    value = value.lower().translate(LEET_TRANSLATION)
    value = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", value)
    value = re.sub(r"[_\-./\\|*~`'\"()\[\]{}:;,+?<>]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


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
        "ibang pasyente",
        "lahat ng pasyente",
        "lahat ng patient",
        "records ng pasyente",
        "patient records",
        "registros de pacientes",
    ]
    return any(_term_present(normalized, term) for term in patient_scope_terms)


def _term_present(normalized, term):
    clean = normalize_security_text(term)
    if not clean:
        return False
    if " " in clean:
        return clean in normalized
    return re.search(rf"\b{re.escape(clean)}\b", normalized) is not None


def _confidence(issues, signals):
    if not issues:
        return 0.0
    score = 0.55 + min(0.4, 0.1 * len(issues)) + min(0.05, 0.01 * len(signals))
    return round(min(score, 0.99), 3)
