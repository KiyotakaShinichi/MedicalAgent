"""
LLM-as-judge for RAG groundedness + hallucination scoring.

Architecture and intent
-----------------------
The existing heuristic in ``backend.services.agent_rag``
(:func:`answer_grounding_score`, :func:`hallucination_score`) compares
overlapping content tokens between the assistant reply and the retrieved
context.  That is fast, deterministic, and cheap — but it cannot tell whether
the *meaning* of the answer is actually supported by the retrieved evidence.
It will happily score a "yes" answer to "is metastasis present?" as well
grounded if both reply and retrieved text mention the word "metastasis".

This module adds a second, independent scoring path: an LLM judge that
evaluates ``(question, context, answer)`` triples and returns

::

    {
      "groundedness":  0..1,
      "hallucination": 0..1,
      "verdict":       "supported" | "partially_supported" | "unsupported",
      "rationale":     "<one short sentence>"
    }

Design choices
~~~~~~~~~~~~~~
* **Opt-in.** Default behavior is unchanged — the eval pipeline still runs the
  heuristic.  Set ``ONCOTRACK_RAG_JUDGE=on`` (or pass ``llm_judge_enabled=True``
  to :func:`judge_rag_answer`) to activate it.  This keeps CI and the existing
  quality-gate runtime predictable.
* **Heuristic is preserved as ``metric_v1``.** The judge becomes ``metric_v2``
  inside the eval report so reviewers can see both side-by-side rather than a
  silent swap.
* **No clinical claim.** The judge is *another* heuristic; an LLM scoring an
  LLM is informative for engineering regression but is **not** clinical
  validation.
* **Hard fail-safe.** If the API call fails, the LLM is unconfigured, or the
  response is non-JSON, this module returns ``status="not_computed"`` with a
  reason — never a fake score.

Provider
~~~~~~~~
Uses the same Groq client the rest of the project relies on so we do not
introduce a second LLM dependency.  The model is configurable via
``ONCOTRACK_RAG_JUDGE_MODEL`` and falls back to whatever
:func:`backend.config.get_groq_model` returns.
"""

from __future__ import annotations

import json
import os
from typing import Any

try:
    from groq import Groq
except Exception:  # groq SDK not installed in some test environments
    Groq = None  # type: ignore[assignment]

from backend.config import get_groq_api_key, get_groq_model


JUDGE_SYSTEM_PROMPT = """\
You are an evaluation judge for a retrieval-augmented medical-information assistant.
You will receive:
  - a user QUESTION
  - the CONTEXT chunks that were retrieved
  - the ASSISTANT ANSWER that was produced

Your job is to score, on the basis of the CONTEXT alone, how well the ANSWER
is grounded.  Do NOT use outside knowledge.  Do NOT make up facts.

Return STRICT JSON with this shape and no other text:
{
  "groundedness":  <float 0..1 — how much of the ANSWER is directly supported by CONTEXT>,
  "hallucination": <float 0..1 — how much of the ANSWER asserts facts not in CONTEXT>,
  "verdict": "supported" | "partially_supported" | "unsupported",
  "rationale": "<one short sentence, <= 30 words>"
}

Rules:
- If the ANSWER is a refusal/redirect (e.g. "I cannot diagnose"), set groundedness=1.0 and hallucination=0.0 and verdict="supported" — refusals are not factual claims.
- If the CONTEXT is empty, set verdict="unsupported" and hallucination >= 0.5.
- Be conservative.  Borderline = "partially_supported".
"""


def is_judge_enabled(explicit: bool | None = None) -> bool:
    """Resolve whether the LLM judge should run.

    Precedence:
      1. ``explicit`` argument (when the caller wants to force on/off, e.g. tests)
      2. ``ONCOTRACK_RAG_JUDGE`` env var (``on``, ``1``, ``true`` → enabled)
      3. otherwise disabled
    """
    if explicit is not None:
        return bool(explicit)
    value = os.environ.get("ONCOTRACK_RAG_JUDGE", "").strip().lower()
    return value in {"on", "1", "true", "yes"}


def judge_rag_answer(
    *,
    question: str,
    answer: str,
    context_chunks: list[dict] | None,
    llm_judge_enabled: bool | None = None,
    timeout_s: float = 6.0,
) -> dict[str, Any]:
    """Score ``(question, context_chunks, answer)`` with an LLM judge.

    Always returns a dict.  When the judge is disabled, unconfigured, or
    fails, the dict contains ``status="not_computed"`` and a ``reason`` —
    never a guessed score.
    """
    if not is_judge_enabled(llm_judge_enabled):
        return _not_computed("ONCOTRACK_RAG_JUDGE flag is off")

    if Groq is None:
        return _not_computed("groq SDK not available in this environment")

    api_key = get_groq_api_key()
    if not api_key:
        return _not_computed("GROQ_API_KEY not configured")

    model = os.environ.get("ONCOTRACK_RAG_JUDGE_MODEL") or get_groq_model()

    context_text = _format_context(context_chunks or [])
    user_payload = {
        "QUESTION": (question or "")[:1500],
        "CONTEXT": context_text[:6000],
        "ANSWER":   (answer   or "")[:2500],
    }

    try:
        client = Groq(api_key=api_key, timeout=timeout_s)
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=220,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user",   "content": json.dumps(user_payload)},
            ],
        )
    except Exception as exc:
        return _not_computed(f"judge_request_failed: {exc.__class__.__name__}")

    content = response.choices[0].message.content if response.choices else None
    parsed = _parse_judge_json(content)
    if not parsed:
        return _not_computed("non_json_response")

    groundedness  = _clamp_unit(parsed.get("groundedness"))
    hallucination = _clamp_unit(parsed.get("hallucination"))
    verdict       = str(parsed.get("verdict") or "").strip().lower()
    if verdict not in {"supported", "partially_supported", "unsupported"}:
        verdict = "partially_supported"

    return {
        "status":        "ok",
        "score":         groundedness,
        "groundedness":  groundedness,
        "hallucination": hallucination,
        "verdict":       verdict,
        "rationale":     str(parsed.get("rationale") or "")[:280],
        "method":        f"llm_judge model={model} (engineering proxy, not clinical validation)",
        "model":         model,
    }


def _format_context(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    parts: list[str] = []
    for idx, chunk in enumerate(chunks[:6], start=1):
        if not isinstance(chunk, dict):
            continue
        title = chunk.get("title") or chunk.get("id") or f"source {idx}"
        text = chunk.get("text") or ""
        parts.append(f"[{idx}] {title}: {text}")
    return "\n".join(parts)


def _clamp_unit(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return round(max(0.0, min(1.0, f)), 3)


def _parse_judge_json(content: Any) -> dict[str, Any] | None:
    text = str(content or "").strip()
    if not text:
        return None
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _not_computed(reason: str) -> dict[str, Any]:
    return {
        "status": "not_computed",
        "score": None,
        "groundedness": None,
        "hallucination": None,
        "verdict": None,
        "rationale": None,
        "method": "llm_judge (engineering proxy)",
        "reason": reason,
    }
