"""Deterministic validation for generated RAG outputs."""
from __future__ import annotations

from hr_rag.schemas import RetrievedChunk


def validate_citations(result: dict, retrieved: list[RetrievedChunk]) -> dict:
    """Reject citations that do not identify evidence supplied to the model."""
    valid_sources = {f"{item.chunk.doc_id} {item.chunk.version}" for item in retrieved}
    citations = result.get("cited_sections", [])
    valid_citations = [citation for citation in citations if any(source in citation for source in valid_sources)]
    if citations and len(valid_citations) == len(citations):
        result["cited_sections"] = valid_citations
        return result
    return {
        "answer": "The policy evidence could not be validated against the generated citations. HR review is required before providing a definitive answer.",
        "draft_email": "Dear Employee,\n\nYour question has been escalated to HR for review. An HR representative will follow up with you.\n\nBest regards,\nHR Support Team",
        "cited_sections": valid_citations,
        "confidence": "low",
        "conflict_flag": True,
        "next_action": "escalate_to_hr_review",
    }