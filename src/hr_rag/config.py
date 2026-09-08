"""
Central configuration for the HR RAG pipeline.

Everything here is read from environment variables (via a .env file) so that
switching models, paths, or chunking parameters never requires touching code.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file in the project root, if present.
load_dotenv()

# huggingface_hub defaults to its newer "Xet" storage protocol for downloads,
# which some corporate proxies block even though classic HTTPS downloads from
# huggingface.co work fine. Force the classic path unless already overridden.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# --- Paths -------------------------------------------------------------
# src/hr_rag/config.py -> parents[2] is the project root (rag-chatbot/)
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
VECTOR_INDEX_DIR = ROOT_DIR / os.getenv("VECTOR_INDEX_DIR", "data/vector_index")

# --- Models --------------------------------------------------------------
# GitHub Models, called through its OpenAI-compatible endpoint so we can use
# the standard `openai` SDK. Auth is a GitHub Personal Access Token (classic
# or fine-grained) with "models: read" permission - the same GitHub account
# used for Copilot, no separate signup required.
GITHUB_MODELS_TOKEN = os.getenv("GITHUB_MODELS_TOKEN", "")
GITHUB_MODELS_MODEL = os.getenv("GITHUB_MODELS_MODEL", "openai/gpt-4o-mini")
GITHUB_MODELS_BASE_URL = os.getenv("GITHUB_MODELS_BASE_URL", "https://models.github.ai/inference")

# --- Chunking --------------------------------------------------------------
# Measured in characters. ~800 chars is roughly 150-200 tokens, small enough
# to keep retrieval precise but large enough to keep a policy clause intact.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# --- Retrieval --------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "5"))
