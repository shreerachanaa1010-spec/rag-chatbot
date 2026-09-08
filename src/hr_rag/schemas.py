"""
Pydantic models shared across the pipeline.

Keeping these in one place means every stage (ingest -> chunk -> retrieve ->
generate) agrees on the exact shape of the data, and we get free validation.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PolicyDocument(BaseModel):
    """A single ingested source document, before chunking."""

    doc_id: str  # e.g. "HR-POL-US-014" - stable ID across versions of the same policy
    version: str  # e.g. "v1.0", "v2.0" - lets us detect superseded documents
    region: str  # e.g. "US", "EU", "Global"
    effective_date: str  # ISO date string, e.g. "2023-03-01"
    title: str
    source_file: str  # filename this was parsed from, for traceability
    text: str  # full cleaned plain-text body


class Chunk(BaseModel):
    """A retrieval-sized slice of a PolicyDocument, with its metadata carried along."""

    chunk_id: str  # unique id, e.g. "HR-POL-US-014_v2.0_003"
    doc_id: str
    version: str
    region: str
    effective_date: str
    title: str
    source_file: str
    chunk_index: int
    text: str


class RetrievedChunk(BaseModel):
    """A Chunk plus its similarity distance from a query, returned by the vector store."""

    chunk: Chunk
    distance: float  # lower = more similar (Chroma uses L2/cosine distance)


class DecisionOutput(BaseModel):
    """
    The structured deliverable the LLM must produce for every employee question.

    This is what would get written back into a case-management / ticketing
    system automatically, instead of a human re-typing the chatbot's answer.
    """

    answer: str
    cited_sections: list[str] = Field(
        description="doc_id + version + short excerpt reference for every fact used"
    )
    confidence: Literal["high", "medium", "low"]
    conflict_flag: bool = Field(
        description="True if retrieved chunks include multiple conflicting policy versions"
    )
    next_action: Literal["send_to_employee", "escalate_to_hr_review"]


class PipelineResult(BaseModel):
    """Everything one call to the pipeline produces for a single question."""

    question: str
    decision: DecisionOutput
    draft_email: str
    retrieved: list[RetrievedChunk]
