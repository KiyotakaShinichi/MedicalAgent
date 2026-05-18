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
# AGENT_CACHE_* + SEMANTIC_CACHE_MIN_SIMILARITY moved to
# backend.services.agent_cache and are re-imported lower in this module
# alongside the rest of the cache layer.

# Module-level cache for the merged KB corpus.  Avoids repeated disk reads on
# every pipeline call.  Call _invalidate_kb_cache() after ingesting new chunks.
_KB_CORPUS_CACHE: list | None = None


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


# input_guardrail_check moved to backend.services.agent_input_gate as
# part of the agent_rag.py module split.  Re-exported below.
from backend.services.agent_input_gate import input_guardrail_check  # noqa: F401, E402


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


# exact_cache_check + semantic_cache_check moved to
# backend.services.agent_cache as part of the agent_rag.py module split.
# Re-imported below alongside the rest of the cache layer.


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


# evaluate_rag_response + the per-metric scorers moved to
# backend.services.agent_eval_scoring as part of the agent_rag.py module
# split.  Re-imported below.


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

# _content_tokens / _estimate_tokens / _score_status / _cost_latency_note
# moved to backend.services.agent_eval_scoring.  Re-imported below.
from backend.services.agent_eval_scoring import (  # noqa: F401, E402
    _content_tokens,
    _cost_latency_note,
    _estimate_tokens,
    _maybe_run_llm_judge,
    _score_status,
    answer_grounding_score,
    estimate_token_and_cost,
    evaluate_rag_response,
    hallucination_score,
    proxy_retrieval_precision_at_k,
)


# is_cacheable + store_cache + cache policy/freshness helpers moved to
# backend.services.agent_cache (re-imported below alongside the lookups).


# _safety_reply moved to backend.services.agent_answer_composition
# (re-imported via _safety_reply alias below for back-compat).
from backend.services.agent_answer_composition import _safety_reply  # noqa: F401, E402


# _security_block_reply moved to backend.services.agent_input_gate.
from backend.services.agent_input_gate import _security_block_reply  # noqa: F401, E402


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


# _mark_cache_hit + _cache_rejection_reason moved to
# backend.services.agent_cache (re-imported below).


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


# Cache layer + cache-adjacent utilities now live in
# backend.services.agent_cache.  Re-import the full surface so existing
# in-module references AND external callers via agent_rag keep working.
from backend.services.agent_cache import (  # noqa: F401, E402
    AGENT_CACHE_SCHEMA_VERSION,
    AGENT_CACHE_TTL_DAYS,
    SEMANTIC_CACHE_MIN_SIMILARITY,
    _cache_policy_snapshot,
    _cache_rejection_reason,
    _cache_response_payload,
    _cache_row_freshness,
    _cache_row_policy,
    _coerce_utc,
    _datetime_to_iso,
    _json_loads,
    _mark_cache_hit,
    _query_hash,
    exact_cache_check,
    is_cacheable,
    semantic_cache_check,
    store_cache,
)
