"""Deterministic safe responses for non-ALLOW evidence dispositions."""

from __future__ import annotations

from typing import Any, Mapping

from backend.services.rag_evidence.types import EvidenceDisposition


def build_safe_abstention(
    disposition: EvidenceDisposition,
    *,
    query: str = "",
    input_guardrails: Mapping[str, Any] | None = None,
    existing_reply: str = "",
) -> str:
    """Return a category-specific English or Taglish safe response."""

    taglish = _is_taglish(query)
    if disposition is EvidenceDisposition.BLOCK_SAFETY:
        if existing_reply and _independent_safety_trigger(input_guardrails):
            return existing_reply
        return (
            "Hindi ko maipagpapatuloy ang request na iyon. Matutulungan kita sa sarili mong NLCare records at support questions."
            if taglish
            else "I can't continue with that request. I can help with your own NLCare records and support questions."
        )
    if disposition is EvidenceDisposition.BLOCK_MEDICAL_BOUNDARY:
        return (
            "Hindi ako makakapagbigay ng diagnosis, prognosis, dose, o treatment decision. Dalhin ito sa oncology care team mo para sa patient-specific review."
            if taglish
            else "I can't provide a diagnosis, prognosis, dose, or treatment decision. Please bring this to your oncology care team for patient-specific review."
        )
    if disposition is EvidenceDisposition.ABSTAIN_INSUFFICIENT_EVIDENCE:
        return (
            "Hindi sapat ang na-verify na source support para masagot ko ito nang ligtas. Matutulungan kitang ayusin ang tanong para sa oncology care team mo."
            if taglish
            else "I couldn't find enough verified source support to answer that safely. I can help you organize the question for your oncology care team."
        )
    if disposition is EvidenceDisposition.ABSTAIN_CONFLICTING_EVIDENCE:
        return (
            "Hindi sapat ang pagkakatugma ng available sources para makagawa ako ng ligtas na buod. Paki-review ito sa oncology care team mo."
            if taglish
            else "The available sources are not consistent enough for me to summarize safely. Please review this with your oncology care team."
        )
    if disposition is EvidenceDisposition.ABSTAIN_UNSUPPORTED_CLAIMS:
        return (
            "Hindi na-verify ng sources ang lahat ng kailangang claims, kaya hindi ko ilalabas ang sagot. Maaari nating gawing tanong ito para sa care team mo."
            if taglish
            else "The sources did not verify every required claim, so I won't release the answer. I can help turn this into a question for your care team."
        )
    return (
        "Hindi ko nakumpleto ang evidence checks na kailangan para sa ligtas na sagot ngayon. Subukan ulit mamaya o itanong sa oncology care team mo."
        if taglish
        else "I couldn't complete the evidence checks needed for a safe answer right now. Please try again later or ask your oncology care team."
    )


def _independent_safety_trigger(input_guardrails: Mapping[str, Any] | None) -> bool:
    if not isinstance(input_guardrails, Mapping):
        return False
    return input_guardrails.get("status") == "failed" or str(input_guardrails.get("level") or "") in {
        "high_risk",
        "blocked",
    }


def _is_taglish(query: str) -> bool:
    lower = f" {str(query or '').lower()} "
    tokens = (
        " hindi ", " wala ", " bakit ", " paano ", " pwede ", " ko ", " mo ",
        " ako ", " kaya ", " ba ", " gamot ", " sagot ",
    )
    return sum(token in lower for token in tokens) >= 2
