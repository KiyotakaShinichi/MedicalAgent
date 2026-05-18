import hashlib
import importlib.util
import json
import os
import re
from time import perf_counter
from datetime import datetime, timezone, timedelta

from backend.models import AgentResponseCache, RAGEvaluationLog
from backend.services.kb_ingestion import load_ingested_chunks
from backend.services.local_llm import configured_llm_providers, decide_cache_with_local_llm, route_intent_with_local_llm
from backend.services.rag_vector_index import corpus_fingerprint, search_hybrid_index
from backend.services.pii_redaction import redact_text
from backend.services.request_context import get_request_id
from backend.services.security_guardrails import detect_multilingual_medical_danger, detect_prompt_injection_or_exfiltration, normalize_security_text


# MAX_CONTEXT_CHARS + _CROSS_ENCODER_CACHE moved to
# backend.services.agent_retrieval and are re-imported lower in this
# module so existing references still resolve.
AGENT_CACHE_TTL_DAYS = 30

# Module-level cache for the merged KB corpus.  Avoids repeated disk reads on
# every pipeline call.  Call _invalidate_kb_cache() after ingesting new chunks.
_KB_CORPUS_CACHE: list | None = None
AGENT_CACHE_SCHEMA_VERSION = "agent_response_cache_v4"
SEMANTIC_CACHE_MIN_SIMILARITY = 0.86


KNOWLEDGE_SNIPPETS = [
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


def run_patient_agent_pipeline(db, patient_id, query, patient_context, fallback_response, actions=None, urgent_flags=None, preselected_intent=None):
    started = perf_counter()
    actions = actions or []
    urgent_flags = urgent_flags or []
    safety = safety_scope_check(query, urgent_flags)
    input_guardrails = input_guardrail_check(query, safety)
    t_safety = perf_counter()
    if input_guardrails["status"] == "failed":
        safety = {
            **safety,
            "level": "high_risk",
            "scope": input_guardrails["scope"],
            "cache_allowed": False,
            "message": input_guardrails["message"],
        }
        intent = "security_boundary"
        rewritten = rewrite_and_decompose(query, intent)
        result = {
            "reply": _security_block_reply(input_guardrails),
            "citations": [],
            "intent": intent,
            "safety": safety,
            "retrieval_context": [],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "safety_note": "Security boundary: the assistant cannot reveal private records, system instructions, database contents, secrets, or raw internal knowledge base data.",
            "validation": {
                "status": "passed",
                "issues": [],
                "citation_count": 0,
            },
            "cache": {
                "status": "blocked_by_input_guardrail",
                "cacheable": False,
                "reason": input_guardrails["scope"],
            },
            "pipeline_trace": _trace(safety, intent, rewritten, [], [], [], "input_guardrail_block"),
        }
        return _finalize_result(
            db=db,
            patient_id=patient_id,
            query=query,
            rewritten=rewritten,
            result=result,
            retrieved=[],
            reranked=[],
            compressed=[],
            input_guardrails=input_guardrails,
            started=started,
        )
    intent = _validated_preselected_intent(preselected_intent, safety) or route_intent(query, actions, safety)
    rewritten = rewrite_and_decompose(query, intent)
    t_routing = perf_counter()
    cacheable = is_cacheable(query, intent, safety, actions, urgent_flags)
    knowledge_fingerprint = knowledge_base_fingerprint()
    cache_policy = _cache_policy_snapshot(knowledge_fingerprint)

    cache_hit = None
    if cacheable:
        cache_hit = exact_cache_check(
            db,
            rewritten["normalized_query"],
            intent=intent,
            safety_level=safety.get("level"),
            knowledge_fingerprint=knowledge_fingerprint,
        )
        if cache_hit is None:
            cache_hit = semantic_cache_check(db, rewritten["semantic_key"], intent, knowledge_fingerprint=knowledge_fingerprint)
    if cache_hit:
        result = {
            **cache_hit["response"],
            "cache": {
                "status": cache_hit["status"],
                "cache_id": cache_hit["cache_id"],
                "cacheable": True,
                "expires_at": cache_hit.get("expires_at"),
                "knowledge_fingerprint": cache_hit.get("knowledge_fingerprint"),
                "policy": cache_hit.get("policy"),
            },
            "pipeline_trace": _trace(safety, intent, rewritten, [], [], [], "cache_hit", cache_policy=cache_policy),
        }
        return _finalize_result(
            db=db,
            patient_id=patient_id,
            query=query,
            rewritten=rewritten,
            result=result,
            retrieved=[],
            reranked=[],
            compressed=result.get("retrieval_context") or [],
            input_guardrails=input_guardrails,
            started=started,
        )

    if _uses_direct_support_lane(intent, safety):
        generated = generate_answer(
            query=query,
            fallback_response=fallback_response,
            safety=safety,
            intent=intent,
            compressed_context=[],
            actions=actions,
            patient_context=patient_context,
        )
        validated = validate_answer_and_citations(generated, [], safety)
        result = {
            **validated,
            "cache": {
                "status": "not_cacheable",
                "cacheable": False,
                "reason": f"intent_not_cacheable:{intent}",
                "policy": cache_policy,
            },
            "pipeline_trace": _trace(safety, intent, rewritten, [], [], [], "direct_support", cache_policy=cache_policy),
        }
        return _finalize_result(
            db=db,
            patient_id=patient_id,
            query=query,
            rewritten=rewritten,
            result=result,
            retrieved=[],
            reranked=[],
            compressed=[],
            input_guardrails=input_guardrails,
            started=started,
        )

    retrieved = hybrid_retrieval(rewritten, intent)
    expanded = expand_parent_child_windows(retrieved)
    t_retrieval = perf_counter()
    reranked = rerank_context(expanded, rewritten, intent, safety)
    compressed = contextual_compression(reranked)
    t_rerank = perf_counter()
    generated = generate_answer(
        query=query,
        fallback_response=fallback_response,
        safety=safety,
        intent=intent,
        compressed_context=compressed,
        actions=actions,
        patient_context=patient_context,
    )
    validated = validate_answer_and_citations(generated, compressed, safety)
    t_generation = perf_counter()

    if cacheable and validated["validation"]["status"] == "passed":
        cache_row = store_cache(db, rewritten, intent, safety, validated, knowledge_fingerprint=knowledge_fingerprint)
        cache_status = {
            "status": "stored",
            "cache_id": cache_row.id,
            "cacheable": True,
            "expires_at": _datetime_to_iso(cache_row.expires_at),
            "knowledge_fingerprint": cache_row.knowledge_fingerprint,
            "policy": cache_policy,
        }
    else:
        cache_status = {
            "status": "not_cacheable",
            "cacheable": False,
            "reason": _cache_rejection_reason(query, intent, safety, actions, urgent_flags),
            "policy": cache_policy,
        }

    stage_ms = {
        "safety_gate_ms": round((t_safety - started) * 1000, 2),
        "intent_routing_ms": round((t_routing - t_safety) * 1000, 2),
        "retrieval_ms": round((t_retrieval - t_routing) * 1000, 2),
        "rerank_ms": round((t_rerank - t_retrieval) * 1000, 2),
        "generation_ms": round((t_generation - t_rerank) * 1000, 2),
    }
    result = {
        **validated,
        "cache": cache_status,
        "pipeline_trace": {
            **_trace(safety, intent, rewritten, retrieved, reranked, compressed, "generated", cache_policy=cache_policy),
            "stage_ms": stage_ms,
        },
    }
    return _finalize_result(
        db=db,
        patient_id=patient_id,
        query=query,
        rewritten=rewritten,
        result=result,
        retrieved=retrieved,
        reranked=reranked,
        compressed=compressed,
        input_guardrails=input_guardrails,
        started=started,
    )


def input_guardrail_check(query, safety):
    security = detect_prompt_injection_or_exfiltration(query)
    lower = query.lower()
    issues = []
    if security["blocked"]:
        issues.extend(security["issues"])
    if safety.get("level") == "high_risk":
        issues.append(safety.get("scope") or "high_risk_medical_scope")

    blocking_issues = {
        "prompt_injection_or_jailbreak",
        "database_or_file_access_attempt",
        "sensitive_data_exfiltration_attempt",
        "privacy_boundary_request",
    }
    status = "failed" if any(issue in blocking_issues for issue in issues) else "passed"
    if status == "failed":
        scope = "input_guardrail_block"
        message = security["message"]
    else:
        scope = safety.get("scope")
        message = "Input guardrail passed."
    return {
        "status": status,
        "scope": scope,
        "issues": sorted(set(issues)),
        "message": message,
        "security": {
            "confidence": security["confidence"],
            "signals": security["signals"],
        },
    }


# safety_scope_check moved to backend.services.agent_safety as part of the
# agent_rag.py module split.  Re-exported here so existing imports
# (chat, eval scripts, tests) keep working unchanged.
from backend.services.agent_safety import safety_scope_check  # noqa: F401, E402


# route_intent + the three conversation detectors moved to
# backend.services.agent_intent_router.  Re-exported so existing imports
# keep working.
from backend.services.agent_intent_router import (  # noqa: F401, E402
    _is_conversation_opening,
    _is_identity_or_capability_question,
    _is_social_checkin,
    route_intent,
)


def _validated_preselected_intent(intent, safety):
    allowed = {
        "safety_boundary",
        "treatment_decision_boundary",
        "data_entry_confirmation",
        "portal_help",
        "patient_timeline_monitoring",
        "education",
        "emotional_support",
        "general_support",
        "conversation",
        "patient_memory",
    }
    if intent not in allowed:
        return None
    if safety.get("scope") == "treatment_decision_request":
        return "treatment_decision_boundary"
    if safety.get("scope") in {"urgent_or_safety_related", "diagnosis_or_outcome_claim"}:
        return "safety_boundary"
    return intent


# _uses_direct_support_lane moved to backend.services.agent_answer_composition
# as part of the agent_rag.py module split.  Re-imported below alongside
# the rest of the answer-composition module.


# rewrite_and_decompose moved to backend.services.agent_query_rewriting as
# part of the agent_rag.py module split.  Re-exported so existing imports
# keep working.
from backend.services.agent_query_rewriting import rewrite_and_decompose  # noqa: F401, E402


def exact_cache_check(db, normalized_query, intent=None, safety_level=None, knowledge_fingerprint=None, now=None):
    knowledge_fingerprint = knowledge_fingerprint or knowledge_base_fingerprint()
    query_hash = _query_hash(normalized_query)
    row = db.query(AgentResponseCache).filter(AgentResponseCache.query_hash == query_hash).first()
    if not row:
        return None
    if intent is not None and row.intent != intent:
        return None
    if safety_level is not None and row.safety_level != safety_level:
        return None
    freshness = _cache_row_freshness(row, knowledge_fingerprint, now=now)
    if freshness["status"] != "fresh":
        return None
    response = _json_loads(row.response_json)
    if response is None:
        return None
    _mark_cache_hit(db, row, now=now)
    return {
        "status": "exact_cache_hit",
        "cache_id": row.id,
        "response": response,
        "expires_at": _datetime_to_iso(row.expires_at),
        "knowledge_fingerprint": row.knowledge_fingerprint,
        "policy": _cache_row_policy(row),
    }


def semantic_cache_check(db, semantic_key, intent, min_similarity=SEMANTIC_CACHE_MIN_SIMILARITY, knowledge_fingerprint=None, now=None):
    knowledge_fingerprint = knowledge_fingerprint or knowledge_base_fingerprint()
    query_tokens = set(semantic_key.split())
    if not query_tokens:
        return None
    rows = (
        db.query(AgentResponseCache)
        .filter(AgentResponseCache.intent == intent)
        .filter(AgentResponseCache.safety_level == "low_risk")
        .filter(AgentResponseCache.knowledge_fingerprint == knowledge_fingerprint)
        .all()
    )
    best = None
    for row in rows:
        freshness = _cache_row_freshness(row, knowledge_fingerprint, now=now)
        if freshness["status"] != "fresh":
            continue
        row_tokens = set((row.semantic_key or "").split())
        if not row_tokens:
            continue
        score = len(query_tokens & row_tokens) / len(query_tokens | row_tokens)
        if score >= min_similarity and (best is None or score > best[0]):
            best = (score, row)
    if best is None:
        return None
    row = best[1]
    response = _json_loads(row.response_json)
    if response is None:
        return None
    _mark_cache_hit(db, row, now=now)
    response["semantic_cache_similarity"] = round(best[0], 3)
    return {
        "status": "semantic_cache_hit",
        "cache_id": row.id,
        "response": response,
        "expires_at": _datetime_to_iso(row.expires_at),
        "knowledge_fingerprint": row.knowledge_fingerprint,
        "policy": _cache_row_policy(row),
    }


# hybrid_retrieval + expand_parent_child_windows moved to
# backend.services.agent_retrieval as part of the agent_rag.py module
# split.  Re-imported below alongside the rest of the retrieval module.


def _knowledge_snippets():
    global _KB_CORPUS_CACHE
    if _KB_CORPUS_CACHE is None:
        _KB_CORPUS_CACHE = KNOWLEDGE_SNIPPETS + load_ingested_chunks()
    return _KB_CORPUS_CACHE


def _invalidate_kb_cache():
    """Call after ingesting new KB chunks so the next pipeline call reloads."""
    global _KB_CORPUS_CACHE
    _KB_CORPUS_CACHE = None


def get_rag_corpus():
    return _knowledge_snippets()


def knowledge_base_fingerprint():
    return corpus_fingerprint(_knowledge_snippets())


# rerank_context, contextual_compression, _CURATED_SOURCES,
# _cross_encoder_* helpers moved to backend.services.agent_retrieval as
# part of the agent_rag.py module split.  Re-imported below alongside
# the rest of the retrieval module.
from backend.services.agent_retrieval import (  # noqa: F401, E402
    CURATED_SOURCES as _CURATED_SOURCES,
    MAX_CONTEXT_CHARS,
    _cross_encoder_enabled,
    _cross_encoder_scores,
    _get_cross_encoder,
    _reranker_backend,
    contextual_compression,
    expand_parent_child_windows,
    hybrid_retrieval,
    rerank_context,
)


# generate_answer, validate_answer_and_citations, REFUSAL_INTENTS, and
# their helpers moved to backend.services.agent_answer_composition.
# Re-imported so existing imports + the few in-module references keep
# working.
from backend.services.agent_answer_composition import (  # noqa: F401, E402
    REFUSAL_INTENTS,
    _uses_direct_support_lane,
    generate_answer,
    validate_answer_and_citations,
)


# _apply_post_gen_validator + _apply_intent_aware_rag_layer moved to
# backend.services.agent_post_gen as part of the agent_rag.py module
# split.  Re-imported so _finalize_result keeps calling them by name.
from backend.services.agent_post_gen import (  # noqa: F401, E402
    _apply_intent_aware_rag_layer,
    _apply_post_gen_validator,
)


def _finalize_result(db, patient_id, query, rewritten, result, retrieved, reranked, compressed, input_guardrails, started):
    """Orchestrate the post-generation pipeline:

      1. Compute latency.
      2. Run legacy output-guardrail heuristics.
      3. Run the post-gen safety validator (may rewrite the reply).
      4. Run the intent-aware RAG layer (mode → tier filter → claim
         validation → evidence grade → optional insufficient-evidence
         substitution).
      5. Build the RAG evaluation telemetry block.
      6. Persist the RAGEvaluationLog row.

    Each step lives in a named helper so the failure surface is explicit
    and the call site reads top-to-bottom.
    """
    latency_ms = round((perf_counter() - started) * 1000, 2)
    output_guardrails = output_guardrail_check(result)
    output_guardrails, pgv_decision = _apply_post_gen_validator(result, output_guardrails)
    _apply_intent_aware_rag_layer(result, retrieved, input_guardrails, pgv_decision)

    rag_evaluation = evaluate_rag_response(
        query=query,
        rewritten=rewritten,
        result=result,
        retrieved=retrieved,
        reranked=reranked,
        compressed=compressed,
        input_guardrails=input_guardrails,
        output_guardrails=output_guardrails,
        latency_ms=latency_ms,
    )
    result["guardrails"] = {
        "input": input_guardrails,
        "output": output_guardrails,
    }
    result["rag_evaluation"] = rag_evaluation
    _store_rag_evaluation_log(
        db=db,
        patient_id=patient_id,
        query=query,
        result=result,
        rag_evaluation=rag_evaluation,
        retrieved=retrieved,
        compressed=compressed,
    )
    return result


def output_guardrail_check(result):
    reply = result.get("reply") or ""
    validation = result.get("validation") or {}
    issues = list(validation.get("issues") or [])
    intent = result.get("intent")
    unsafe_terms = [
        "you should stop",
        "you should start",
        "increase your dose",
        "decrease your dose",
        "skip chemo",
        "you are cancer free",
        "you have metastasis",
    ]
    if any(term in reply.lower() for term in unsafe_terms):
        issues.append("unsafe_output_directive_or_diagnosis")
    # On refusal intents, citations are intentionally stripped; see
    # generate_answer. The missing-citations check would otherwise fire on
    # every safety_boundary / treatment_decision_boundary reply that surfaces
    # background education context for context-only display.
    if (
        intent not in REFUSAL_INTENTS
        and (result.get("retrieval_context") or [])
        and not (result.get("citations") or [])
    ):
        issues.append("missing_citations")
    safety = result.get("safety") or {}
    if safety.get("level") == "high_risk" and not any(term in reply.lower() for term in ["oncology", "emergency", "clinician", "care team"]):
        issues.append("missing_high_risk_escalation")
    return {
        "status": "passed" if not issues else "failed",
        "issues": sorted(set(issues)),
    }


def evaluate_rag_response(query, rewritten, result, retrieved, reranked, compressed, input_guardrails, output_guardrails, latency_ms):
    retrieval_precision = proxy_retrieval_precision_at_k(reranked or retrieved, rewritten, k=3)
    grounding = answer_grounding_score(result.get("reply") or "", compressed)
    hallucination = hallucination_score(
        grounding_score=grounding["score"],
        validation=result.get("validation") or {},
        input_guardrails=input_guardrails,
        output_guardrails=output_guardrails,
        citations=result.get("citations") or [],
        compressed=compressed,
    )

    # Opt-in second scoring path: LLM-as-judge.  The heuristic above stays
    # as `answer_grounding` / `hallucination` (metric_v1).  When the judge
    # is enabled it adds `answer_grounding_v2_llm_judge` so a reviewer can
    # see both side-by-side instead of a silent swap.
    judge_result = _maybe_run_llm_judge(query, result.get("reply") or "", compressed)

    token_cost = estimate_token_and_cost(query, result.get("reply") or "", compressed)
    payload = {
        "retrieval_precision_at_3": retrieval_precision,
        "answer_grounding": grounding,
        "hallucination": hallucination,
        "cost_latency": {
            **token_cost,
            "latency_ms": latency_ms,
            "cache_status": (result.get("cache") or {}).get("status"),
            "tradeoff_note": _cost_latency_note((result.get("cache") or {}).get("status"), latency_ms, token_cost["estimated_total_tokens"]),
        },
        "guardrail_summary": {
            "input_status": input_guardrails.get("status"),
            "output_status": output_guardrails.get("status"),
            "input_issues": input_guardrails.get("issues") or [],
            "output_issues": output_guardrails.get("issues") or [],
        },
        "metric_limitations": (
            "Retrieval precision and `answer_grounding`/`hallucination` are heuristic token-overlap proxies. "
            "`answer_grounding_v2_llm_judge`, when present, is an LLM-as-judge second opinion — also a proxy, not clinical validation."
        ),
    }
    if judge_result is not None:
        payload["answer_grounding_v2_llm_judge"] = judge_result
    return payload


def _maybe_run_llm_judge(query, reply, compressed):
    """Run the LLM judge if enabled.  Returns None when disabled so the
    eval payload stays small in default runs.  Imports are local because
    the judge is opt-in and we don't want to pull in groq on import."""
    try:
        from backend.services.llm_judge import is_judge_enabled, judge_rag_answer
    except Exception:
        return None
    if not is_judge_enabled():
        return None
    try:
        return judge_rag_answer(
            question=query,
            answer=reply,
            context_chunks=compressed,
        )
    except Exception as exc:
        # Never break the pipeline on a judge failure; record the reason.
        return {
            "status": "not_computed",
            "reason": f"judge_unexpected_failure: {exc.__class__.__name__}",
            "method": "llm_judge (engineering proxy)",
        }


def proxy_retrieval_precision_at_k(items, rewritten, k=3):
    top = (items or [])[:k]
    if not top:
        return {
            "metric": "proxy_retrieval_precision_at_3",
            "value": None,
            "k": k,
            "relevant_count": 0,
            "method": "No retrieved context.",
            "status": "unavailable",
        }
    query_tokens = set(_tokenize(rewritten.get("expanded_query") or ""))
    relevant_count = 0
    for item in top:
        item_tokens = set(_tokenize(" ".join([
            item.get("title", ""),
            item.get("text", ""),
            " ".join(item.get("tags", [])),
        ])))
        if query_tokens & item_tokens:
            relevant_count += 1
    value = round(relevant_count / len(top), 3)
    return {
        "metric": "proxy_retrieval_precision_at_3",
        "value": value,
        "k": len(top),
        "relevant_count": relevant_count,
        "method": "Heuristic query-token overlap with retrieved source title/tags/text. Replace with labeled precision@k or RAGAS context precision later.",
        "status": _score_status(value, strong=0.8, acceptable=0.6),
    }


def answer_grounding_score(reply, compressed):
    if not reply:
        return {"score": 0.0, "status": "failed", "method": "Empty reply."}
    if not compressed:
        return {
            "score": None,
            "status": "unavailable",
            "method": "No retrieved context; answer may be deterministic fallback rather than RAG-grounded.",
        }
    reply_tokens = set(_content_tokens(reply))
    context_tokens = set()
    for item in compressed:
        context_tokens.update(_content_tokens(item.get("text", "")))
        context_tokens.update(_content_tokens(item.get("title", "")))
    if not reply_tokens:
        score = 0.0
    else:
        score = len(reply_tokens & context_tokens) / len(reply_tokens)
    score = round(score, 3)
    return {
        "score": score,
        "status": _score_status(score, strong=0.55, acceptable=0.35),
        "method": "Heuristic content-token overlap between answer and retrieved context. Upgrade to RAGAS faithfulness/answer relevancy later.",
    }


def hallucination_score(grounding_score, validation, input_guardrails, output_guardrails, citations, compressed):
    issues = set(validation.get("issues") or [])
    issues.update(input_guardrails.get("issues") or [])
    issues.update(output_guardrails.get("issues") or [])
    if grounding_score is None:
        base = 0.25 if not compressed else 0.5
    else:
        base = max(0.0, 1.0 - grounding_score)
    if compressed and not citations:
        base += 0.25
    if issues:
        base += min(0.45, 0.15 * len(issues))
    score = round(min(1.0, base), 3)
    if score <= 0.35:
        risk = "low"
    elif score <= 0.65:
        risk = "medium"
    else:
        risk = "high"
    return {
        "score": score,
        "risk": risk,
        "method": "Heuristic inverse grounding plus citation and guardrail penalties. Replace/compare with RAGAS faithfulness later.",
        "issues": sorted(issues),
    }


def estimate_token_and_cost(query, reply, compressed):
    context_chars = sum(len(item.get("text", "")) for item in compressed)
    input_tokens = _estimate_tokens(query) + _estimate_tokens(" ".join(item.get("text", "") for item in compressed))
    output_tokens = _estimate_tokens(reply)
    total_tokens = input_tokens + output_tokens
    return {
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_total_tokens": total_tokens,
        "estimated_context_chars": context_chars,
        "estimated_llm_cost_usd": 0.0,
        "cost_basis": "Current agent path is deterministic/local. Token estimates are logged for future LLM/RAGAS cost analysis.",
    }


def _store_rag_evaluation_log(db, patient_id, query, result, rag_evaluation, retrieved, compressed):
    hallucination = rag_evaluation["hallucination"]
    grounding = rag_evaluation["answer_grounding"]
    retrieval_precision = rag_evaluation["retrieval_precision_at_3"]
    cost_latency = rag_evaluation["cost_latency"]
    guardrails = rag_evaluation["guardrail_summary"]
    row = RAGEvaluationLog(
        patient_id=patient_id,
        request_id=get_request_id(),
        query_hash=_query_hash(_normalize_query(query)),
        query_preview=redact_text(str(query or ""))[:120],
        intent=result.get("intent") or "unknown",
        safety_level=(result.get("safety") or {}).get("level") or "unknown",
        cache_status=(result.get("cache") or {}).get("status"),
        terminal_step=(result.get("pipeline_trace") or {}).get("terminal_step"),
        retrieval_precision_at_3=retrieval_precision.get("value"),
        grounding_score=grounding.get("score"),
        hallucination_score=hallucination.get("score"),
        hallucination_risk=hallucination.get("risk"),
        input_guardrail_status=guardrails.get("input_status"),
        output_guardrail_status=guardrails.get("output_status"),
        latency_ms=cost_latency.get("latency_ms"),
        estimated_input_tokens=cost_latency.get("estimated_input_tokens"),
        estimated_output_tokens=cost_latency.get("estimated_output_tokens"),
        estimated_total_tokens=cost_latency.get("estimated_total_tokens"),
        estimated_llm_cost_usd=cost_latency.get("estimated_llm_cost_usd"),
        retrieved_source_ids_json=json.dumps([item.get("id") for item in retrieved if item.get("id")]),
        cited_source_ids_json=json.dumps([item.get("id") for item in result.get("citations") or []]),
        guardrail_issues_json=json.dumps({
            "input": guardrails.get("input_issues") or [],
            "output": guardrails.get("output_issues") or [],
        }),
        rag_mode=result.get("rag_mode"),
        rewritten_query=result.get("rewritten_query"),
        evidence_grade_json=json.dumps(result.get("evidence_grade")) if result.get("evidence_grade") is not None else None,
        claim_validation_json=json.dumps(result.get("claim_validation")) if result.get("claim_validation") is not None else None,
        tier_filter_json=json.dumps(result.get("tier_filter")) if result.get("tier_filter") is not None else None,
        post_gen_validator_json=json.dumps(result.get("post_gen_validator")) if result.get("post_gen_validator") is not None else None,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


# _contains_diagnostic_or_treatment_claim moved to
# backend.services.agent_answer_composition (re-imported above).

def _content_tokens(text):
    generic = {
        "general", "information", "portal", "patient", "team", "care", "use", "discuss", "personal",
        "decisions", "oncology", "medical", "review", "contact", "emergency", "services", "support",
        "assistant", "tracking", "education", "only",
    }
    return [token for token in _tokenize(text) if token not in generic and len(token) > 2]


def _estimate_tokens(text):
    return max(1, int(len(text or "") / 4))


def _score_status(value, strong, acceptable):
    if value is None:
        return "unavailable"
    if value >= strong:
        return "strong"
    if value >= acceptable:
        return "acceptable"
    return "unideal"


def _cost_latency_note(cache_status, latency_ms, total_tokens):
    if cache_status in {"exact_cache_hit", "semantic_cache_hit"}:
        return "Cache hit: lower latency and no new retrieval/generation cost."
    if total_tokens > 800 or latency_ms > 1500:
        return "Generated path is heavier; consider caching if this is low-risk and reusable."
    return "Generated path is within current PoC latency/token budget."


def is_cacheable(query, intent, safety, actions=None, urgent_flags=None):
    actions = actions or []
    urgent_flags = urgent_flags or []
    lower = query.lower()
    patient_specific_terms = [" my ", " me ", " i ", "latest", "my score", "my labs", "my mri", "my treatment"]
    if actions or urgent_flags or not safety.get("cache_allowed"):
        return False
    if intent not in {"education", "portal_help", "general_support"}:
        return False
    padded = f" {lower} "
    if any(term in padded for term in patient_specific_terms):
        return False
    llm = decide_cache_with_local_llm(query, deterministic_cacheable=True, intent=intent, safety=safety)
    if llm.get("available") and float(llm.get("confidence") or 0) >= 0.72:
        return bool(llm.get("cacheable"))
    return True


def store_cache(db, rewritten, intent, safety, response, knowledge_fingerprint=None, now=None):
    now = now or datetime.now(timezone.utc)
    knowledge_fingerprint = knowledge_fingerprint or knowledge_base_fingerprint()
    query_hash = _query_hash(rewritten["normalized_query"])
    row = db.query(AgentResponseCache).filter(AgentResponseCache.query_hash == query_hash).first()
    if row is None:
        row = AgentResponseCache(query_hash=query_hash)
        db.add(row)
    else:
        row.hit_count = 0
        row.last_hit_at = None

    row.semantic_key = rewritten["semantic_key"]
    row.intent = intent
    row.safety_level = safety["level"]
    row.normalized_query = rewritten["normalized_query"]
    row.response_json = json.dumps(_cache_response_payload(response), default=str)
    row.source_ids_json = json.dumps([item["id"] for item in response.get("citations") or []])
    row.knowledge_fingerprint = knowledge_fingerprint
    row.cache_schema_version = AGENT_CACHE_SCHEMA_VERSION
    row.cache_policy_json = json.dumps(_cache_policy_snapshot(knowledge_fingerprint), default=str)
    row.expires_at = now + timedelta(days=AGENT_CACHE_TTL_DAYS)
    row.updated_at = now
    db.commit()
    db.refresh(row)
    return row


def _cache_response_payload(response):
    return {
        "reply": response.get("reply"),
        "citations": response.get("citations") or [],
        "intent": response.get("intent"),
        "safety": response.get("safety"),
        "retrieval_context": response.get("retrieval_context") or [],
        "generated_at": response.get("generated_at"),
        "safety_note": response.get("safety_note"),
        "validation": response.get("validation"),
    }


def _cache_policy_snapshot(knowledge_fingerprint):
    return {
        "schema_version": AGENT_CACHE_SCHEMA_VERSION,
        "ttl_days": AGENT_CACHE_TTL_DAYS,
        "semantic_min_similarity": SEMANTIC_CACHE_MIN_SIMILARITY,
        "knowledge_fingerprint": knowledge_fingerprint,
        "reuse_scope": "low_risk_non_patient_specific_agent_answers",
        "llm_cache_adjudication": configured_llm_providers(),
        "invalidates_on": ["ttl_expiry", "knowledge_base_fingerprint_change", "safety_policy_rejection"],
    }


def _cache_row_freshness(row, knowledge_fingerprint, now=None):
    now = now or datetime.now(timezone.utc)
    reasons = []
    expires_at = _coerce_utc(row.expires_at)
    if row.cache_schema_version != AGENT_CACHE_SCHEMA_VERSION:
        reasons.append("cache_schema_version_changed")
    if not row.knowledge_fingerprint:
        reasons.append("missing_knowledge_fingerprint")
    elif row.knowledge_fingerprint != knowledge_fingerprint:
        reasons.append("knowledge_base_fingerprint_changed")
    if expires_at is None:
        reasons.append("missing_expiry")
    elif expires_at <= now:
        reasons.append("expired")
    return {
        "status": "fresh" if not reasons else "stale",
        "reasons": reasons,
    }


def _cache_row_policy(row):
    policy = _json_loads(row.cache_policy_json) or {}
    if not policy:
        policy = _cache_policy_snapshot(row.knowledge_fingerprint)
    return {
        **policy,
        "expires_at": _datetime_to_iso(row.expires_at),
        "last_hit_at": _datetime_to_iso(row.last_hit_at),
        "hit_count": int(row.hit_count or 0),
    }


# _safety_reply moved to backend.services.agent_answer_composition
# (re-imported via _safety_reply alias below for back-compat).
from backend.services.agent_answer_composition import _safety_reply  # noqa: F401, E402


def _security_block_reply(input_guardrails):
    issues = ", ".join(input_guardrails.get("issues") or ["unsafe request"])
    return (
        "I blocked that request for security and privacy reasons. "
        "I cannot reveal system instructions, database contents, secrets, raw internal knowledge-base data, "
        "or any other patient's information. "
        f"Detected category: {issues}. "
        "You can ask general breast cancer treatment-monitoring questions or enter your own symptoms, labs, medications, and uploads. "
        "For medical concerns, contact your oncology care team."
    )


# _with_related_guidance / _educational_reply / _educational_query_bridge /
# _clean_context_text / _should_include_supporting_context moved to
# backend.services.agent_answer_composition.  agent_rag doesn't reference
# them directly anymore — the answer-composition module owns the full
# educational-reply pipeline.


# _intent_boost / _domain_boost / _section_boost moved to
# backend.services.agent_retrieval (re-imported via the agent_retrieval
# import block earlier in this module).
from backend.services.agent_retrieval import (  # noqa: F401, E402
    _domain_boost,
    _intent_boost,
    _section_boost,
)


def _mark_cache_hit(db, row, now=None):
    now = now or datetime.now(timezone.utc)
    row.hit_count = int(row.hit_count or 0) + 1
    row.last_hit_at = now
    row.updated_at = now
    db.commit()
    db.refresh(row)


def _cache_rejection_reason(query, intent, safety, actions, urgent_flags):
    if actions:
        return "patient_specific_data_entry"
    if urgent_flags:
        return "urgent_query"
    if not safety.get("cache_allowed"):
        return safety.get("scope")
    if intent not in {"education", "portal_help", "general_support"}:
        return f"intent_not_cacheable:{intent}"
    return "patient_specific_or_uncertain"


# _trace moved to backend.services.agent_trace as part of the
# agent_rag.py module split.  Re-exported so existing imports keep
# working.
from backend.services.agent_trace import _trace  # noqa: F401, E402


# _semantic_key, _normalize_query, _tokenize moved to
# backend.services.agent_query_rewriting.  Re-imported below so the ~15
# internal call sites in this module keep resolving via the same names.
from backend.services.agent_query_rewriting import (  # noqa: E402
    _normalize_query,
    _semantic_key,
    _tokenize,
)


def _query_hash(normalized_query):
    return hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()


def _coerce_utc(value):
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _datetime_to_iso(value):
    value = _coerce_utc(value)
    return value.isoformat() if value else None


def _json_loads(value):
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None
