"""FastAPI entry point for the HR policy resolution engine."""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hr_rag.config import VECTOR_INDEX_DIR
from hr_rag.email_service import send_hr_email
from hr_rag.pipeline import ask


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    region: str | None = Field(default=None, max_length=32)


class AskResponse(BaseModel):
    question: str
    answer: str
    draft_email: str
    confidence: str
    conflict_flag: bool
    next_action: str
    cited_sections: list[str]
    retrieved: list[dict]
    older_version_warning: str | None = None


class HREmailRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    region: str | None = Field(default=None, max_length=32)
    answer: str = Field(min_length=1, max_length=10000)
    draft_email: str = Field(min_length=1, max_length=10000)
    cited_sections: list[str] = Field(default_factory=list, max_length=50)


def _is_question_like(value: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", value)
    return len(words) >= 2 and (len(words) >= 3 or "?" in value)


app = FastAPI(
    title="HR Policy Resolution API",
    version="0.1.0",
    description="Grounded HR policy answers with citations and review flags.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str | bool]:
    index_ready = (VECTOR_INDEX_DIR / "policy_faiss" / "index.faiss").exists()
    return {"status": "ok", "index_ready": index_ready}


@app.get("/api/index")
def index_status() -> dict[str, str | bool]:
    index_ready = (VECTOR_INDEX_DIR / "policy_faiss" / "index.faiss").exists()
    return {
        "ready": index_ready,
        "location": str(VECTOR_INDEX_DIR),
        "message": "Index is ready" if index_ready else "Run scripts/02_build_index.py first",
    }


@app.post("/api/ask", response_model=AskResponse)
def ask_policy(request: AskRequest) -> AskResponse:
    question = request.question.strip()
    if not _is_question_like(question):
        raise HTTPException(
            status_code=400,
            detail="Please enter a specific HR policy question, including the relevant policy area or location.",
        )

    if not (VECTOR_INDEX_DIR / "policy_faiss" / "index.faiss").exists():
        raise HTTPException(status_code=503, detail="Vector index is not ready")

    try:
        result = ask(question, region=request.region or None)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Unable to generate policy answer: {exc}") from exc

    return AskResponse(
        question=result.question,
        answer=result.decision.answer,
        draft_email=result.draft_email,
        confidence=result.decision.confidence,
        conflict_flag=result.decision.conflict_flag,
        next_action=result.decision.next_action,
        cited_sections=result.decision.cited_sections,
        retrieved=[item.model_dump() for item in result.retrieved],
        older_version_warning=result.older_version_warning,
    )


@app.post("/api/send-to-hr")
def send_to_hr(request: HREmailRequest) -> dict[str, str]:
    subject = f"HR policy review required: {request.question[:80]}"
    citations = "\n".join(f"- {citation}" for citation in request.cited_sections)
    body = (
        "HR policy review requested\n\n"
        f"Employee question: {request.question}\n"
        f"Region: {request.region or 'Not specified'}\n\n"
        f"Current assessment:\n{request.answer}\n\n"
        f"Draft employee response:\n{request.draft_email}\n\n"
        f"Policy references:\n{citations or '- None returned'}\n"
    )
    try:
        send_hr_email(subject, body)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Unable to send the HR email.") from exc
    return {"status": "sent", "message": "The case was sent to HR for review."}


_FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
