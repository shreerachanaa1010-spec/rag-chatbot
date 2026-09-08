"""
Step 4a: Embeddings (TF-IDF).

Turns text into vectors using TF-IDF (term frequency - inverse document
frequency): a classic sparse-vector representation from traditional
information retrieval. No neural network, no model download, fully offline -
this sidesteps the corporate network block on Hugging Face entirely, and is
genuinely how retrieval worked before dense neural embeddings existed (and
still what powers e.g. Elasticsearch/BM25 in production today).

Each chunk becomes a vector where every dimension is a word (or word pair,
via ngram_range) from the corpus vocabulary, weighted by how distinctive
that term is to that chunk versus the rest of the corpus.

Because the vocabulary and IDF weights are *learned from the corpus* (via
`.fit()`), the same fitted vectorizer must be reused for both indexing and
querying - you can't independently embed one piece of text in isolation the
way a neural embedding model can. So we fit once at index-build time and
persist the fitted vectorizer alongside the vector index (see vectorstore.py).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def fit_vectorizer(texts: list[str]) -> TfidfVectorizer:
    """Learn a vocabulary + IDF weights from the given corpus of chunk texts."""
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.95)
    vectorizer.fit(texts)
    return vectorizer


def embed_with_vectorizer(vectorizer: TfidfVectorizer, texts: list[str]) -> sp.csr_matrix:
    """Transform text into TF-IDF vectors, L2-normalized so dot product == cosine similarity."""
    matrix = vectorizer.transform(texts)
    return normalize(matrix)


def save_vectorizer(vectorizer: TfidfVectorizer, path: Path) -> None:
    path.write_bytes(pickle.dumps(vectorizer))


def load_vectorizer(path: Path) -> TfidfVectorizer:
    return pickle.loads(path.read_bytes())

