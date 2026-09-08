"""
Step 5 + 6: Retrieval and conflict detection.

`retrieve()` is a thin pass-through to the vector store (kept separate so the
pipeline doesn't need to know about Chroma directly).

`detect_version_conflicts()` is a deterministic, code-level safety net: if
the top-K retrieved chunks include more than one *version* of the same
`doc_id` (e.g., the 2021 and 2023 parental leave policies both got pulled
in because the question is ambiguous or a policy was recently amended), we
must not let the LLM silently pick one. We detect that here in plain Python
so the pipeline can force an escalation regardless of what the LLM says.
"""
from __future__ import annotations

from collections import defaultdict

from hr_rag.config import TOP_K
from hr_rag.schemas import RetrievedChunk
from hr_rag.vectorstore import query as vector_query


def retrieve(question: str, region: str | None = None, top_k: int = TOP_K) -> list[RetrievedChunk]:
    return vector_query(question, top_k=top_k, region=region)


def detect_version_conflicts(retrieved: list[RetrievedChunk]) -> dict[str, set[str]]:
    """
    Returns {doc_id: {version, ...}} only for doc_ids where 2+ distinct
    versions appear among the retrieved chunks.
    """
    versions_by_doc: dict[str, set[str]] = defaultdict(set)
    for rc in retrieved:
        versions_by_doc[rc.chunk.doc_id].add(rc.chunk.version)

    return {doc_id: versions for doc_id, versions in versions_by_doc.items() if len(versions) > 1}
