"""
Step 2 + 3 runner: parse data/raw/ -> PolicyDocuments -> Chunks, and save both
to data/processed/ as JSON so later steps (embedding) can be run independently
without re-parsing documents every time.

Usage:
    python scripts/01_ingest.py
"""
from __future__ import annotations

import json

from rich.console import Console
from rich.table import Table

from hr_rag.chunking import chunk_documents
from hr_rag.config import PROCESSED_DATA_DIR
from hr_rag.ingest import load_documents, save_documents

console = Console()


def main() -> None:
    console.print("[bold]Step 2: Ingesting documents from data/raw/...[/bold]")
    documents = load_documents()

    doc_table = Table(title="Ingested Documents")
    for col in ["doc_id", "version", "region", "effective_date", "source_file"]:
        doc_table.add_column(col)
    for d in documents:
        doc_table.add_row(d.doc_id, d.version, d.region, d.effective_date, d.source_file)
    console.print(doc_table)

    save_documents(documents)
    console.print(f"[green]Saved {len(documents)} documents -> data/processed/documents.json[/green]\n")

    console.print("[bold]Step 3: Chunking documents...[/bold]")
    chunks = chunk_documents(documents)

    chunk_table = Table(title="Chunking Summary")
    chunk_table.add_column("doc_id")
    chunk_table.add_column("version")
    chunk_table.add_column("num_chunks")
    counts: dict[tuple[str, str], int] = {}
    for c in chunks:
        counts[(c.doc_id, c.version)] = counts.get((c.doc_id, c.version), 0) + 1
    for (doc_id, version), count in counts.items():
        chunk_table.add_row(doc_id, version, str(count))
    console.print(chunk_table)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / "chunks.json"
    out_path.write_text(json.dumps([c.model_dump() for c in chunks], indent=2), encoding="utf-8")
    console.print(f"[green]Saved {len(chunks)} chunks -> {out_path}[/green]")


if __name__ == "__main__":
    main()
