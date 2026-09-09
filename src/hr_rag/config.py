"""
Central configuration for the HR RAG pipeline.

Everything here is read from environment variables (via a .env file) so that
switching models, paths, or chunking parameters never requires touching code.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Resolve the project root before loading .env so startup works from any cwd.
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

# huggingface_hub defaults to its newer "Xet" storage protocol for downloads,
# which some corporate proxies block even though classic HTTPS downloads from
# huggingface.co work fine. Force the classic path unless already overridden.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# --- Paths -------------------------------------------------------------
# src/hr_rag/config.py -> parents[2] is the project root (rag-chatbot/)
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
VECTOR_INDEX_DIR = ROOT_DIR / os.getenv("VECTOR_INDEX_DIR", "data/vector_index")

# --- Models --------------------------------------------------------------
# Gemini exposes an OpenAI-compatible endpoint, so the existing OpenAI SDK
# can be used without changing the pipeline's response contract.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_BASE_URL = os.getenv(
	"GEMINI_BASE_URL",
	"https://generativelanguage.googleapis.com/v1beta/openai/",
)
EMBEDDING_MODEL = os.getenv(
	"EMBEDDING_MODEL",
	"sentence-transformers/all-MiniLM-L6-v2",
)
RERANKER_MODEL = os.getenv(
	"RERANKER_MODEL",
	"cross-encoder/ms-marco-MiniLM-L-6-v2",
)

# --- Chunking --------------------------------------------------------------
# Measured in characters. ~800 chars is roughly 150-200 tokens, small enough
# to keep retrieval precise but large enough to keep a policy clause intact.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# --- Retrieval --------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "1.6"))
