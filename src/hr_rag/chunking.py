"""
Step 3: Chunking.

Splits each PolicyDocument's text into overlapping, retrieval-sized Chunks.

Why chunk at all? Embedding models and the LLM's context window both work
best on small, focused pieces of text. If we embedded a whole 2-page policy
as one vector, a question about "how many weeks of leave" would retrieve the
*entire* document (imprecise) instead of just the "Leave Entitlement"
section (precise).

Why overlap? If a sentence describing an important rule happens to fall
right on a chunk boundary, a naive non-overlapping split could cut it in
half, so neither chunk contains the full rule. Overlap (CHUNK_OVERLAP chars
repeated at the start of the next chunk) makes that much less likely, at the
cost of a bit of redundant storage.

Strategy used here: split on paragraph breaks first (keeps whole sentences/
clauses intact), then greedily pack paragraphs into windows of ~CHUNK_SIZE
characters, carrying the tail of one window into the start of the next.
This is simple, dependency-free, and good enough for structured policy docs.
More advanced strategies (semantic chunking, sentence-window, recursive
splitters) build on this same idea.
"""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from hr_rag.config import CHUNK_OVERLAP, CHUNK_SIZE
from hr_rag.schemas import Chunk, PolicyDocument


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split policy text with LangChain's structure-aware recursive splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def chunk_document(document: PolicyDocument) -> list[Chunk]:
    """Chunk a single PolicyDocument into a list of Chunk objects."""
    text_chunks = chunk_text(document.text)
    chunks: list[Chunk] = []

    for i, text in enumerate(text_chunks):
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}_{document.version}_{i:03d}",
                doc_id=document.doc_id,
                version=document.version,
                region=document.region,
                effective_date=document.effective_date,
                title=document.title,
                source_file=document.source_file,
                chunk_index=i,
                text=text,
            )
        )

    return chunks


def chunk_documents(documents: list[PolicyDocument]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for document in documents:
        all_chunks.extend(chunk_document(document))
    return all_chunks
