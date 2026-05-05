import hashlib
import json
import re
from time import perf_counter
from datetime import datetime, timezone

from backend.models import AgentResponseCache, RAGEvaluationLog
from backend.services.kb_ingestion import load_ingested_chunks
from backend.services.security_guardrails import detect_prompt_injection_or_exfiltration


MAX_CONTEXT_CHARS = 1300


KNOWLEDGE_SNIPPETS = [
    {
        "id": "cdc-fever-chemo",
        "parent_id": "infection-safety",
        "title": "Fever during chemotherapy",
        "source_name": "CDC",
        "source_url": "https://www.cdc.gov/cancer-preventing-infections/patients/fever.html",
        "tags": ["fever", "infection", "chemotherapy", "urgent", "wbc", "neutropenia"],
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
        "text": (
            "Breast cancer chemotherapy may be given before surgery to shrink tumor burden or after surgery to reduce recurrence risk. "
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
        "text": (
            "In this PoC, pCR means pathologic complete response, a treatment-response label used in some breast cancer research datasets. "
            "The project treats it as a classification target, not as a diagnosis or patient-facing clinical conclusion."
        ),
    },
    {
        "id": "project-monitoring-score",
        "parent_id": "response-modeling",
        "title": "Monitoring score boundary",
        "source_name": "Project safety policy",
        "source_url": "README.md",
        "tags": ["score", "probability", "model", "response", "monitoring", "classification"],
        "text": (
            "The treatment monitoring score is an exploratory engineering signal that combines model response signals with CBC and symptom concerns. "
            "It is for trend discussion and clinician review, not a treatment decision."
        ),
    },
    {
        "id": "portal-upload-guide",
        "parent_id": "portal-help",
        "title": "What patients can upload",
        "source_name": "Project patient portal guide",
        "source_url": "README.md",
        "tags": ["upload", "portal", "cbc", "mri", "symptoms", "medications", "labs"],
        "text": (
            "The patient portal is designed to store CBC/lab values, MRI or imaging files, imaging report text, medications, treatments, "
            "and symptoms so changes can be summarized over time."
        ),
    },
]


def run_patient_agent_pipeline(db, patient_id, query, patient_context, fallback_response, actions=None, urgent_flags=None):
    started = perf_counter()
    actions = actions or []
    urgent_flags = urgent_flags or []
    safety = safety_scope_check(query, urgent_flags)
    input_guardrails = input_guardrail_check(query, safety)
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
    intent = route_intent(query, actions, safety)
    rewritten = rewrite_and_decompose(query, intent)
    cacheable = is_cacheable(query, intent, safety, actions, urgent_flags)

    cache_hit = None
    if cacheable:
        cache_hit = exact_cache_check(db, rewritten["normalized_query"])
        if cache_hit is None:
            cache_hit = semantic_cache_check(db, rewritten["semantic_key"], intent)
    if cache_hit:
        result = {
            **cache_hit["response"],
            "cache": {
                "status": cache_hit["status"],
                "cache_id": cache_hit["cache_id"],
                "cacheable": True,
            },
            "pipeline_trace": _trace(safety, intent, rewritten, [], [], [], "cache_hit"),
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

    retrieved = hybrid_retrieval(rewritten, intent)
    expanded = expand_parent_child_windows(retrieved)
    reranked = rerank_context(expanded, rewritten, intent, safety)
    compressed = contextual_compression(reranked)
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

    if cacheable and validated["validation"]["status"] == "passed":
        cache_row = store_cache(db, rewritten, intent, safety, validated)
        cache_status = {"status": "stored", "cache_id": cache_row.id, "cacheable": True}
    else:
        cache_status = {"status": "not_cacheable", "cacheable": False, "reason": _cache_rejection_reason(query, intent, safety, actions, urgent_flags)}

    result = {
        **validated,
        "cache": cache_status,
        "pipeline_trace": _trace(safety, intent, rewritten, retrieved, reranked, compressed, "generated"),
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


def safety_scope_check(query, urgent_flags=None):
    lower = query.lower()
    urgent_flags = urgent_flags or []
    decision_terms = [
        "should i stop",
        "should i start",
        "should i change",
        "what dose",
        "increase my dose",
        "decrease my dose",
        "skip chemo",
        "skip treatment",
    ]
    diagnostic_terms = [
        "do i have cancer",
        "is it metastatic",
        "am i cancer free",
        "is my cancer gone",
        "diagnose me",
    ]
    if urgent_flags or any(term in lower for term in ["fever", "chest pain", "cannot breathe", "bleeding", "suicidal", "self harm"]):
        return {
            "level": "high_risk",
            "scope": "urgent_or_safety_related",
            "cache_allowed": False,
            "message": "Urgent or safety-related wording detected; answer must route toward clinician/emergency review.",
        }
    if any(term in lower for term in decision_terms):
        return {
            "level": "high_risk",
            "scope": "treatment_decision_request",
            "cache_allowed": False,
            "message": "Treatment decision wording detected; assistant must not recommend medication or treatment changes.",
        }
    if any(term in lower for term in diagnostic_terms):
        return {
            "level": "high_risk",
            "scope": "diagnosis_or_outcome_claim",
            "cache_allowed": False,
            "message": "Diagnosis/outcome confirmation wording detected; assistant must not confirm disease state.",
        }
    return {
        "level": "low_risk",
        "scope": "education_or_tracking",
        "cache_allowed": True,
        "message": "Low-risk educational or portal-support query.",
    }


def route_intent(query, actions=None, safety=None):
    lower = query.lower()
    actions = actions or []
    safety = safety or {}
    if safety.get("scope") == "treatment_decision_request":
        return "treatment_decision_boundary"
    if safety.get("scope") in {"urgent_or_safety_related", "diagnosis_or_outcome_claim"}:
        return "safety_boundary"
    if actions:
        return "data_entry_confirmation"
    if any(term in lower for term in ["upload", "site", "portal", "dashboard", "where can i", "how do i add"]):
        return "portal_help"
    if any(term in lower for term in ["last 14", "timeline", "cycle", "toxicity", "score", "my treatment", "working", "progress"]):
        return "patient_timeline_monitoring"
    if any(term in lower for term in ["pcr", "response", "mri", "cbc", "wbc", "hemoglobin", "platelets", "chemo", "chemotherapy", "side effect"]):
        return "education"
    if any(term in lower for term in ["anxious", "worried", "sad", "scared", "depressed"]):
        return "emotional_support"
    return "general_support"


def rewrite_and_decompose(query, intent):
    normalized = _normalize_query(query)
    expanded = normalized
    synonyms = {
        "wbc": "white blood cells cbc infection neutropenia",
        "hgb": "hemoglobin anemia cbc",
        "hb": "hemoglobin anemia cbc",
        "plt": "platelets cbc bleeding",
        "mri": "imaging response breast mri",
        "pcr": "pathologic complete response treatment response classification",
        "chemo": "chemotherapy treatment side effects",
    }
    for term, expansion in synonyms.items():
        if term in normalized.split():
            expanded = f"{expanded} {expansion}"
    parts = [
        part.strip()
        for part in re.split(r"\band\b|\?|,", normalized)
        if part.strip()
    ]
    if not parts:
        parts = [normalized]
    return {
        "original_query": query,
        "normalized_query": normalized,
        "expanded_query": expanded,
        "subqueries": parts[:4],
        "semantic_key": _semantic_key(expanded, intent),
    }


def exact_cache_check(db, normalized_query):
    query_hash = _query_hash(normalized_query)
    row = db.query(AgentResponseCache).filter(AgentResponseCache.query_hash == query_hash).first()
    if not row:
        return None
    _mark_cache_hit(db, row)
    return {
        "status": "exact_cache_hit",
        "cache_id": row.id,
        "response": _json_loads(row.response_json),
    }


def semantic_cache_check(db, semantic_key, intent, min_similarity=0.86):
    query_tokens = set(semantic_key.split())
    if not query_tokens:
        return None
    rows = (
        db.query(AgentResponseCache)
        .filter(AgentResponseCache.intent == intent)
        .filter(AgentResponseCache.safety_level == "low_risk")
        .all()
    )
    best = None
    for row in rows:
        row_tokens = set((row.semantic_key or "").split())
        if not row_tokens:
            continue
        score = len(query_tokens & row_tokens) / len(query_tokens | row_tokens)
        if score >= min_similarity and (best is None or score > best[0]):
            best = (score, row)
    if best is None:
        return None
    row = best[1]
    _mark_cache_hit(db, row)
    response = _json_loads(row.response_json)
    response["semantic_cache_similarity"] = round(best[0], 3)
    return {
        "status": "semantic_cache_hit",
        "cache_id": row.id,
        "response": response,
    }


def hybrid_retrieval(rewritten, intent):
    query_tokens = set(_tokenize(rewritten["expanded_query"]))
    rows = []
    for snippet in _knowledge_snippets():
        text_tokens = set(_tokenize(" ".join([snippet["title"], snippet["text"], " ".join(snippet["tags"])])))
        lexical = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
        semantic = len(query_tokens & set(snippet["tags"])) / max(len(set(snippet["tags"])), 1)
        intent_boost = _intent_boost(intent, snippet)
        score = lexical + semantic + intent_boost
        if score > 0:
            rows.append({
                **snippet,
                "retrieval_score": round(score, 4),
                "matched_terms": sorted(query_tokens & text_tokens)[:10],
            })
    return sorted(rows, key=lambda row: row["retrieval_score"], reverse=True)[:5]


def expand_parent_child_windows(retrieved):
    seen = {item["id"] for item in retrieved}
    expanded = list(retrieved)
    parent_ids = {item["parent_id"] for item in retrieved}
    for snippet in _knowledge_snippets():
        if snippet["parent_id"] in parent_ids and snippet["id"] not in seen:
            expanded.append({
                **snippet,
                "retrieval_score": 0.15,
                "matched_terms": [],
                "expansion": "parent_child_window",
            })
            seen.add(snippet["id"])
    return expanded


def _knowledge_snippets():
    external = load_ingested_chunks()
    return KNOWLEDGE_SNIPPETS + external


def rerank_context(expanded, rewritten, intent, safety):
    query_tokens = set(_tokenize(rewritten["expanded_query"]))
    reranked = []
    for item in expanded:
        tags = set(item["tags"])
        coverage = len(query_tokens & tags)
        safety_boost = 0.4 if safety.get("level") == "high_risk" and "urgent" in tags else 0
        source_boost = 0.2 if item["source_name"] in {"CDC", "National Cancer Institute", "American Cancer Society"} else 0.05
        final_score = float(item.get("retrieval_score", 0)) + coverage * 0.18 + safety_boost + source_boost
        reranked.append({**item, "rerank_score": round(final_score, 4)})
    return sorted(reranked, key=lambda row: row["rerank_score"], reverse=True)[:5]


def contextual_compression(reranked):
    compressed = []
    total = 0
    for item in reranked:
        text = item["text"]
        if total + len(text) > MAX_CONTEXT_CHARS and compressed:
            continue
        compressed.append({
            "id": item["id"],
            "title": item["title"],
            "source_name": item["source_name"],
            "source_url": item["source_url"],
            "text": text,
            "score": item.get("rerank_score", item.get("retrieval_score")),
        })
        total += len(text)
        if len(compressed) >= 3:
            break
    return compressed


def generate_answer(query, fallback_response, safety, intent, compressed_context, actions, patient_context):
    citations = [
        {
            "id": item["id"],
            "title": item["title"],
            "source_name": item["source_name"],
            "source_url": item["source_url"],
        }
        for item in compressed_context
    ]
    if safety.get("level") == "high_risk":
        reply = _safety_reply(fallback_response, compressed_context, safety)
    elif actions:
        reply = _with_related_guidance(fallback_response, compressed_context)
    elif intent in {"education", "portal_help", "general_support", "emotional_support"} and compressed_context:
        reply = _educational_reply(query, intent, compressed_context)
    else:
        reply = fallback_response

    return {
        "reply": reply,
        "citations": citations,
        "intent": intent,
        "safety": safety,
        "retrieval_context": compressed_context,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "safety_note": "This assistant provides tracking support and education only. It does not diagnose or choose treatment.",
    }


def validate_answer_and_citations(generated, compressed_context, safety):
    reply = generated.get("reply") or ""
    citations = generated.get("citations") or []
    issues = []
    if compressed_context and not citations:
        issues.append("retrieved_context_without_citations")
    unsafe_terms = ["you should stop", "you should start", "increase your dose", "decrease your dose", "skip chemo"]
    if any(term in reply.lower() for term in unsafe_terms):
        issues.append("treatment_directive_detected")
    if safety.get("level") == "high_risk" and not any(term in reply.lower() for term in ["oncology", "emergency", "clinician", "care team"]):
        issues.append("high_risk_reply_missing_escalation")
    if _contains_diagnostic_or_treatment_claim(reply):
        issues.append("diagnostic_or_treatment_claim_detected")

    status = "passed" if not issues else "failed"
    if issues:
        generated["reply"] = (
            "I cannot safely answer that as a treatment or diagnosis decision. "
            "Please contact your oncology care team for medical review. "
            "If symptoms feel sudden, severe, or unsafe, use local emergency services."
        )
    generated["validation"] = {
        "status": status,
        "issues": issues,
        "citation_count": len(citations),
    }
    return generated


def _finalize_result(db, patient_id, query, rewritten, result, retrieved, reranked, compressed, input_guardrails, started):
    latency_ms = round((perf_counter() - started) * 1000, 2)
    output_guardrails = output_guardrail_check(result)
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
    if (result.get("retrieval_context") or []) and not (result.get("citations") or []):
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
    token_cost = estimate_token_and_cost(query, result.get("reply") or "", compressed)
    return {
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
            "Retrieval precision, grounding, and hallucination are heuristic proxy metrics until a labeled KB and RAGAS evaluation set are added."
        ),
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
        query_hash=_query_hash(_normalize_query(query)),
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
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _contains_diagnostic_or_treatment_claim(reply):
    lower = reply.lower()
    blocked_patterns = [
        "you are cancer free",
        "your cancer is gone",
        "you have metastasis",
        "you do not have metastasis",
        "stop chemotherapy",
        "start chemotherapy",
        "change your dose",
    ]
    return any(pattern in lower for pattern in blocked_patterns)


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
    return True


def store_cache(db, rewritten, intent, safety, response):
    row = AgentResponseCache(
        query_hash=_query_hash(rewritten["normalized_query"]),
        semantic_key=rewritten["semantic_key"],
        intent=intent,
        safety_level=safety["level"],
        normalized_query=rewritten["normalized_query"],
        response_json=json.dumps(_cache_response_payload(response), default=str),
        source_ids_json=json.dumps([item["id"] for item in response.get("citations") or []]),
    )
    db.add(row)
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


def _safety_reply(fallback_response, compressed_context, safety):
    context_text = " ".join(item["text"] for item in compressed_context[:2])
    if context_text:
        return (
            f"{fallback_response} Related safety guidance: {context_text} "
            "For medical decisions or urgent symptoms, contact the oncology care team or emergency services."
        )
    return (
        f"{fallback_response} I cannot safely make diagnosis or treatment decisions. "
        "Please contact the oncology care team for medical review."
    )


def _security_block_reply(input_guardrails):
    issues = ", ".join(input_guardrails.get("issues") or ["unsafe request"])
    return (
        "I blocked that request for security and privacy reasons. "
        "I cannot reveal system instructions, database contents, secrets, raw internal knowledge-base data, "
        "or any other patient's information. "
        f"Detected category: {issues}. "
        "You can ask general breast cancer treatment-monitoring questions or enter your own symptoms, labs, medications, and uploads."
    )


def _with_related_guidance(fallback_response, compressed_context):
    if not compressed_context:
        return fallback_response
    guidance = compressed_context[0]["text"]
    return f"{fallback_response} Related guidance: {guidance}"


def _educational_reply(query, intent, compressed_context):
    primary = compressed_context[0]["text"]
    supporting = compressed_context[1]["text"] if len(compressed_context) > 1 else None
    if intent == "portal_help":
        opener = "For this portal:"
    else:
        opener = "General information:"
    reply = f"{opener} {primary}"
    if supporting:
        reply += f" {supporting}"
    reply += " Use this as education and discuss personal decisions with the oncology team."
    return reply


def _intent_boost(intent, snippet):
    tags = set(snippet["tags"])
    boosts = {
        "portal_help": {"upload", "portal", "labs", "mri"},
        "education": {"pcr", "cbc", "wbc", "chemotherapy", "side effects", "mri"},
        "patient_timeline_monitoring": {"score", "monitoring", "cbc", "response"},
        "safety_boundary": {"urgent", "fever", "infection"},
        "treatment_decision_boundary": {"treatment", "doctor", "chemotherapy"},
        "emotional_support": {"symptoms", "doctor", "side effects"},
    }
    desired = boosts.get(intent, set())
    return 0.25 if tags & desired else 0


def _mark_cache_hit(db, row):
    row.hit_count = int(row.hit_count or 0) + 1
    row.updated_at = datetime.now(timezone.utc)
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


def _trace(safety, intent, rewritten, retrieved, reranked, compressed, terminal_step):
    return {
        "steps": [
            "safety_scope_check",
            "intent_router",
            "query_rewrite_decomposition",
            "exact_cache_check",
            "semantic_cache_check",
            "hybrid_retrieval",
            "parent_child_window_expansion",
            "reranker",
            "contextual_compression",
            "answer_generation",
            "validation_citation_check",
            "safe_cache_store",
        ],
        "terminal_step": terminal_step,
        "safety_level": safety.get("level"),
        "intent": intent,
        "subquery_count": len(rewritten.get("subqueries") or []),
        "retrieved_count": len(retrieved),
        "reranked_count": len(reranked),
        "compressed_count": len(compressed),
    }


def _semantic_key(expanded_query, intent):
    tokens = sorted(set(_tokenize(f"{intent} {expanded_query}")))
    return " ".join(tokens[:40])


def _query_hash(normalized_query):
    return hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()


def _normalize_query(query):
    return " ".join(re.sub(r"[^a-z0-9\s/.-]", " ", query.lower()).split())


def _tokenize(text):
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "can", "do", "for", "from", "how", "i", "in",
        "is", "it", "me", "my", "of", "on", "or", "the", "this", "to", "what", "when", "where",
        "why", "with", "you", "your",
    }
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in stopwords and len(token) > 1]


def _json_loads(value):
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None
