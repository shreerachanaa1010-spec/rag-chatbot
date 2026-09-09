"""
Step 4c runner: load data/processed/chunks.json (produced by 01_ingest.py)
and build a semantic vector index from every chunk.

Usage:
    python scripts/02_build_index.py
"""
from __future__ import annotations

import json

from rich.console import Console

from hr_rag.config import PROCESSED_DATA_DIR
from hr_rag.schemas import Chunk
from hr_rag.vectorstore import index_chunks

console = Console()


def main() -> None:
    chunks_path = PROCESSED_DATA_DIR / "chunks.json"
    if not chunks_path.exists():
        console.print(
            "[red]No chunks.json found. Run `python scripts/01_ingest.py` first.[/red]"
        )
        return

    raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = [Chunk(**c) for c in raw_chunks]

    console.print(f"[bold]Embedding and indexing {len(chunks)} chunks...[/bold]")

    index_chunks(chunks)

    console.print(f"[green]Done. FAISS index persisted to data/vector_index/policy_faiss/.[/green]")


if __name__ == "__main__":
    main()
