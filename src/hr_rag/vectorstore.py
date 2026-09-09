"""FAISS vector store implemented through LangChain's vector-store adapter.

Rather than depend on a full vector database, this is a minimal, transparent
implementation of exactly what a vector store does under the hood:

1. Encode every chunk with the same sentence-transformer model and persist
    normalized dense vectors.
2. Encode the query with that model and compute cosine similarity against
    every row.
3. Return the top-K most similar rows.

Because our vectors are L2-normalized (see embeddings.py), cosine similarity
between two vectors reduces to a plain dot product. So similarity search
across N chunks is just: `matrix @ query_vector`, a single sparse
matrix-vector multiply. Production vector databases (Qdrant, Weaviate,
pgvector, ...) add indexes such as HNSW so this scales without a full scan.
For a few thousand HR policy chunks, the local NumPy implementation remains
fast and easy to reason about.

Persistence: the dense matrix is saved as .npy, with corresponding chunk
metadata in a parallel JSON list. Row i in the matrix always corresponds to
entry i in the metadata list.
"""
from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from hr_rag.config import EMBEDDING_MODEL, MAX_DISTANCE, TOP_K, VECTOR_INDEX_DIR
from hr_rag.schemas import Chunk, RetrievedChunk

_INDEX_NAME = "policy_faiss"


def _embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def _index_path(index_dir: Path = VECTOR_INDEX_DIR) -> Path:
    index_dir.mkdir(parents=True, exist_ok=True)
    return index_dir / _INDEX_NAME


def index_chunks(chunks: list[Chunk], index_dir: Path = VECTOR_INDEX_DIR) -> None:
    """Build and persist a LangChain FAISS index from policy chunks."""
    if not chunks:
        return

    documents = [
        Document(page_content=chunk.text, metadata=chunk.model_dump())
        for chunk in chunks
    ]
    store = FAISS.from_documents(documents, _embeddings())
    store.save_local(str(_index_path(index_dir)))


def _load_index(index_dir: Path = VECTOR_INDEX_DIR) -> FAISS:
    index_path = _index_path(index_dir)
    if not (index_path / "index.faiss").exists():
        raise FileNotFoundError(
            "No vector index found. Run `python scripts/02_build_index.py` first."
        )
    return FAISS.load_local(
        str(index_path),
        _embeddings(),
        allow_dangerous_deserialization=True,
    )


def query(question: str, top_k: int = TOP_K, region: str | None = None) -> list[RetrievedChunk]:
    """
    Embed the question and return the top_k most similar chunks.

    If `region` is given, only chunks tagged with that region (or "Global")
    are considered - our region-aware retrieval guardrail.
    """
    store = _load_index()
    metadata_filter = {"region": {"$in": [region, "Global"]}} if region else None
    matches = store.similarity_search_with_score(
        question,
        k=top_k,
        filter=metadata_filter,
        fetch_k=max(top_k * 4, 20),
    )
    retrieved: list[RetrievedChunk] = []
    for document, distance in matches:
        if float(distance) > MAX_DISTANCE:
            continue
        retrieved.append(
            RetrievedChunk(
                chunk=Chunk(**document.metadata),
                distance=float(distance),
            )
        )
    return retrieved
