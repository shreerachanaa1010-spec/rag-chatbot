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
from datetime import date, datetime
from functools import lru_cache

from sentence_transformers import CrossEncoder

from hr_rag.config import RERANKER_MODEL, TOP_K
from hr_rag.schemas import RetrievedChunk
from hr_rag.vectorstore import query as vector_query


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


def _rerank(question: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    if not candidates:
        return []
    scores = _reranker().predict([(question, item.chunk.text) for item in candidates])
    ranked = sorted(zip(scores, candidates), key=lambda item: float(item[0]), reverse=True)
    return [item for _, item in ranked[:top_k]]


def retrieve(question: str, region: str | None = None, top_k: int = TOP_K) -> list[RetrievedChunk]:
    retrieved = _rerank(
        question,
        vector_query(question, top_k=max(top_k * 3, 10), region=region),
        top_k,
    )
    latest_dates: dict[str, date] = {}
    for item in retrieved:
        effective_date = _parse_effective_date(item.chunk.effective_date)
        latest_dates[item.chunk.doc_id] = max(
            latest_dates.get(item.chunk.doc_id, date.min), effective_date
        )

    # Keep older versions visible for auditability, but rank the active version first.
    return sorted(
        retrieved,
        key=lambda item: (
            _parse_effective_date(item.chunk.effective_date)
            != latest_dates[item.chunk.doc_id],
            item.distance,
        ),
    )


def _parse_effective_date(value: str) -> date:
    for pattern in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, pattern).date()
        except ValueError:
            continue
    return date.min


def current_version_chunks(retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Return only the newest effective version for each policy document."""
    latest_dates: dict[str, date] = {}
    for item in retrieved:
        latest_dates[item.chunk.doc_id] = max(
            latest_dates.get(item.chunk.doc_id, date.min),
            _parse_effective_date(item.chunk.effective_date),
        )
    return [
        item
        for item in retrieved
        if _parse_effective_date(item.chunk.effective_date)
        == latest_dates[item.chunk.doc_id]
    ]


def older_version_warning(retrieved: list[RetrievedChunk]) -> str | None:
    """Describe superseded versions retained in the retrieved audit context."""
    versions_by_doc: dict[str, set[str]] = defaultdict(set)
    for item in retrieved:
        versions_by_doc[item.chunk.doc_id].add(item.chunk.version)
    warnings = [
        f"{doc_id}: versions {', '.join(sorted(versions))} were found; the newest effective version was used."
        for doc_id, versions in versions_by_doc.items()
        if len(versions) > 1
    ]
    return " ".join(warnings) if warnings else None


def detect_version_conflicts(retrieved: list[RetrievedChunk]) -> dict[str, set[str]]:
    """
    Returns {doc_id: {version, ...}} only for doc_ids where 2+ distinct
    versions appear among the retrieved chunks.
    """
    versions_by_doc: dict[str, set[str]] = defaultdict(set)
    for rc in retrieved:
        versions_by_doc[rc.chunk.doc_id].add(rc.chunk.version)

    return {doc_id: versions for doc_id, versions in versions_by_doc.items() if len(versions) > 1}
