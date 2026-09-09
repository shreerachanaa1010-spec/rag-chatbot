from __future__ import annotations

from hr_rag.retrieval import current_version_chunks, older_version_warning, retrieve
from hr_rag.schemas import Chunk, RetrievedChunk


def test_latest_effective_version_is_selected() -> None:
    chunks = [
        RetrievedChunk(
            chunk=Chunk(
                chunk_id="policy_v1_000",
                doc_id="HR-POL-001",
                version="v1.0",
                region="US",
                effective_date="January 1, 2021",
                title="Policy",
                source_file="old.md",
                chunk_index=0,
                text="old policy",
            ),
            distance=0.1,
        ),
        RetrievedChunk(
            chunk=Chunk(
                chunk_id="policy_v2_000",
                doc_id="HR-POL-001",
                version="v2.0",
                region="US",
                effective_date="March 1, 2023",
                title="Policy amendment",
                source_file="new.md",
                chunk_index=0,
                text="new policy",
            ),
            distance=0.2,
        ),
    ]

    current = current_version_chunks(chunks)

    assert [item.chunk.version for item in current] == ["v2.0"]
    assert "v1.0, v2.0" in (older_version_warning(chunks) or "")


def test_compensation_query_retrieves_compensation_policy() -> None:
    results = retrieve("How are salary increases and bonuses decided?", region="US")

    assert results
    assert results[0].chunk.doc_id == "HR-POL-US-040"


def test_performance_query_retrieves_performance_policy() -> None:
    results = retrieve("How does the annual performance review work?", region="US")

    assert results
    assert results[0].chunk.doc_id == "HR-POL-GLOBAL-041"
