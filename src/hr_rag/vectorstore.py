"""
Step 4b: Vector store (built from scratch with TF-IDF + scipy sparse matrices).

Rather than depend on a full vector database, this is a minimal, transparent
implementation of exactly what a vector store does under the hood:

1. Fit a TF-IDF vectorizer on the corpus and store every chunk's vector as a
   row in one big sparse matrix.
2. To search, vectorize the query with the *same* fitted vectorizer and
   compute its similarity against every row.
3. Return the top-K most similar rows.

Because our vectors are L2-normalized (see embeddings.py), cosine similarity
between two vectors reduces to a plain dot product. So similarity search
across N chunks is just: `matrix @ query_vector`, a single sparse
matrix-vector multiply. Production vector databases (Chroma, FAISS,
pgvector, ...) do this same computation, they just add approximate nearest-
neighbor indexes (e.g. HNSW) so it scales to millions of vectors without a
full scan. For a few thousand HR policy chunks, brute force is fast enough
and a lot easier to reason about.

Persistence: the sparse matrix is saved as a .npz file, the corresponding
chunk metadata (region, doc_id, version, text, ...) as a parallel JSON list
(row i in the matrix always corresponds to entry i in the metadata list),
and the *fitted* TfidfVectorizer itself, since queries must be transformed
using the exact same vocabulary/IDF weights the index was built with.
"""
from __future__ import annotations

import json
from pathlib import Path

import scipy.sparse as sp

from hr_rag.config import TOP_K, VECTOR_INDEX_DIR
from hr_rag.embeddings import embed_with_vectorizer, fit_vectorizer, load_vectorizer, save_vectorizer
from hr_rag.schemas import Chunk, RetrievedChunk

_MATRIX_FILE = "embeddings.npz"
_METADATA_FILE = "metadata.json"
_VECTORIZER_FILE = "vectorizer.pkl"


def _paths(index_dir: Path = VECTOR_INDEX_DIR) -> tuple[Path, Path, Path]:
    index_dir.mkdir(parents=True, exist_ok=True)
    return (
        index_dir / _MATRIX_FILE,
        index_dir / _METADATA_FILE,
        index_dir / _VECTORIZER_FILE,
    )


def index_chunks(chunks: list[Chunk], index_dir: Path = VECTOR_INDEX_DIR) -> None:
    """Fit a TF-IDF vectorizer on all chunk texts and persist vectors + metadata to disk."""
    if not chunks:
        return

    texts = [c.text for c in chunks]
    vectorizer = fit_vectorizer(texts)
    matrix = embed_with_vectorizer(vectorizer, texts)
    metadata = [c.model_dump() for c in chunks]

    matrix_path, meta_path, vectorizer_path = _paths(index_dir)
    sp.save_npz(matrix_path, matrix)
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    save_vectorizer(vectorizer, vectorizer_path)


def _load_index(index_dir: Path = VECTOR_INDEX_DIR):
    matrix_path, meta_path, vectorizer_path = _paths(index_dir)
    if not matrix_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "No vector index found. Run `python scripts/02_build_index.py` first."
        )
    matrix = sp.load_npz(matrix_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    vectorizer = load_vectorizer(vectorizer_path)
    return matrix, metadata, vectorizer


def query(question: str, top_k: int = TOP_K, region: str | None = None) -> list[RetrievedChunk]:
    """
    Vectorize the question (using the index's fitted vectorizer) and return
    the top_k most similar chunks.

    If `region` is given, only chunks tagged with that region (or "Global")
    are considered - our region-aware retrieval guardrail.
    """
    matrix, metadata, vectorizer = _load_index()
    query_vector = embed_with_vectorizer(vectorizer, [question])

    # Dot product of normalized vectors == cosine similarity, for every row at once.
    similarities = (matrix @ query_vector.T).toarray().ravel()

    candidate_indices = range(len(metadata))
    if region:
        candidate_indices = [
            i for i in candidate_indices if metadata[i]["region"] in (region, "Global")
        ]

    ranked = sorted(candidate_indices, key=lambda i: similarities[i], reverse=True)
    top_indices = ranked[:top_k]

    retrieved: list[RetrievedChunk] = []
    for i in top_indices:
        chunk = Chunk(**metadata[i])
        # Report as a "distance" (lower = more similar) to keep the concept
        # consistent regardless of which similarity metric is used underneath.
        distance = float(1.0 - similarities[i])
        retrieved.append(RetrievedChunk(chunk=chunk, distance=distance))

    return retrieved
