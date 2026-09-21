"""
Step 2: Ingestion.

Reads every file in data/raw/ and turns it into a PolicyDocument: plain text
plus the metadata (doc_id, version, region, effective_date) we need later for
region-aware retrieval and conflict detection.

Supported formats:
- .md / .txt : our HR docs use a small structured header
    ("**Region:** ...", "**Document ID:** ...", etc.) which we parse with a
    regex. This is a stand-in for whatever structured metadata a real HR
    document management system would give us.
- .pdf       : parsed with pypdf. Real PDFs usually don't have this metadata
    in the text, so we fall back to sensible defaults and rely on a filename
    convention instead.
- .docx      : parsed with python-docx, same fallback behavior as PDF.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)

from hr_rag.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from hr_rag.schemas import PolicyDocument

# Matches lines like "**Region:** United States" -> ("Region", "United States")
_METADATA_LINE = re.compile(r"^\*\*(?P<key>[\w ]+):\*\*\s*(?P<value>.+)$", re.MULTILINE)

# Raw "**Region:**" values are free-text descriptions (e.g. "European Union
# (Germany, France, Netherlands offices)"), but region-aware retrieval needs
# to filter on a stable code. Normalize to short codes here, once, at ingest
# time, rather than repeating fuzzy matching logic at query time.
_REGION_CODES = {
    "united states": "US",
    "european union": "EU",
    "global": "Global",
}
_POLICY_AREAS = {
    "parental_leave": ("parental", "maternity", "paternity"),
    "pto": ("pto", "paid time off", "vacation"),
    "compensation": ("compensation", "salary", "bonus"),
    "performance": ("performance", "annual review", "pip"),
    "health_benefits": ("health insurance", "medical benefits"),
    "remote_work": ("remote work", "hybrid", "work from home"),
    "conduct": ("employee conduct", "harassment", "retaliation"),
    "career_development": ("career development", "learning budget", "mentoring"),
}


def _normalize_region(raw_region: str) -> str:
    lowered = raw_region.lower()
    for keyword, code in _REGION_CODES.items():
        if lowered.startswith(keyword):
            return code
    return raw_region


def _infer_policy_area(text: str) -> str:
    lowered = text.lower()
    return next((area for area, terms in _POLICY_AREAS.items() if any(term in lowered for term in terms)), "general")


def _document_status(raw: str) -> str:
    lowered = raw.lower()
    return "archived" if "archived" in lowered or "superseded" in lowered else "active"


def _parse_markdown(path: Path) -> PolicyDocument:
    raw = TextLoader(str(path), encoding="utf-8").load()[0].page_content

    fields = {m.group("key").strip(): m.group("value").strip() for m in _METADATA_LINE.finditer(raw)}

    # Title is the first line, e.g. "# US Parental Leave Policy"
    title_match = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else path.stem

    # "Document ID" line looks like "HR-POL-US-014 (v2.0) - SUPERSEDES v1.0"
    doc_id_raw = fields.get("Document ID", path.stem)
    doc_id_match = re.match(r"([\w-]+)(?:\s*\((?P<version>v[\d.]+)\))?", doc_id_raw)
    doc_id = doc_id_match.group(1) if doc_id_match else doc_id_raw
    version = doc_id_match.group("version") if doc_id_match and doc_id_match.group("version") else "v1.0"
    supersedes_match = re.search(r"supersedes\s+(v[\d.]+)", doc_id_raw, re.IGNORECASE)

    return PolicyDocument(
        doc_id=doc_id,
        version=version,
        region=_normalize_region(fields.get("Region", "Unknown")),
        effective_date=fields.get("Effective Date", "Unknown"),
        title=title,
        source_file=path.name,
        text=raw,
        policy_area=_infer_policy_area(f"{title} {raw}"),
        employee_types=["full_time"] if "full-time" in raw.lower() else ["all"],
        document_status=_document_status(raw),
        supersedes_version=supersedes_match.group(1) if supersedes_match else None,
    )


def _parse_pdf(path: Path) -> PolicyDocument:
    pages = PyPDFLoader(str(path)).load()
    text = "\n".join(page.page_content for page in pages)
    return PolicyDocument(
        doc_id=path.stem,
        version="v1.0",
        region="Unknown",
        effective_date="Unknown",
        title=path.stem,
        source_file=path.name,
        text=text,
        policy_area=_infer_policy_area(f"{path.stem} {text}"),
    )


def _parse_docx(path: Path) -> PolicyDocument:
    text = "\n".join(page.page_content for page in Docx2txtLoader(str(path)).load())
    return PolicyDocument(
        doc_id=path.stem,
        version="v1.0",
        region="Unknown",
        effective_date="Unknown",
        title=path.stem,
        source_file=path.name,
        text=text,
        policy_area=_infer_policy_area(f"{path.stem} {text}"),
    )


_PARSERS = {
    ".md": _parse_markdown,
    ".txt": _parse_markdown,
    ".pdf": _parse_pdf,
    ".docx": _parse_docx,
}


def load_documents(raw_dir: Path = RAW_DATA_DIR) -> list[PolicyDocument]:
    """Parse every supported file in raw_dir into a list of PolicyDocument."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw document directory does not exist: {raw_dir}")
    documents: list[PolicyDocument] = []
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file():
            continue
        parser = _PARSERS.get(path.suffix.lower())
        if parser is None:
            continue
        documents.append(parser(path))
    validate_documents(documents)
    return documents


def validate_documents(documents: list[PolicyDocument]) -> None:
    """Fail closed when source metadata is incomplete or duplicated."""
    if not documents:
        raise ValueError("No supported policy documents were found")
    seen: set[tuple[str, str]] = set()
    for document in documents:
        key = (document.doc_id, document.version)
        if key in seen:
            raise ValueError(f"Duplicate policy version: {document.doc_id} {document.version}")
        seen.add(key)
        if not document.text.strip() or not document.doc_id.strip() or not document.version.strip():
            raise ValueError(f"Missing required metadata or text in {document.source_file}")


def save_documents(documents: list[PolicyDocument], out_dir: Path = PROCESSED_DATA_DIR) -> Path:
    validate_documents(documents)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "documents.json"
    temporary_path = out_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps([d.model_dump() for d in documents], indent=2), encoding="utf-8")
    os.replace(temporary_path, out_path)
    return out_path
