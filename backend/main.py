"""FastAPI entry point for the HR policy resolution engine."""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hr_rag.config import ALLOWED_ORIGINS, API_KEY, REVIEW_DB_PATH, VECTOR_INDEX_DIR
from hr_rag.email_service import send_hr_email
from hr_rag.observability import increment, metrics_snapshot
from hr_rag.pipeline import ask
from hr_rag.review_store import ReviewStore


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
    search_criteria: dict | None = None


class ChatHistoryResponse(BaseModel):
    chats: list[dict]


class HREmailRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    region: str | None = Field(default=None, max_length=32)
    answer: str = Field(min_length=1, max_length=10000)
    draft_email: str = Field(min_length=1, max_length=10000)
    cited_sections: list[str] = Field(default_factory=list, max_length=50)


class ReviewUpdate(BaseModel):
    status: str = Field(pattern="^(open|in_review|resolved|rejected)$")
    reviewer_note: str | None = Field(default=None, max_length=5000)


def _is_question_like(value: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", value)
    return len(words) >= 2 and (len(words) >= 3 or "?" in value)


app = FastAPI(
    title="HR Policy Resolution API",
    version="0.1.0",
    description="Grounded HR policy answers with citations and review flags.",
)
review_store = ReviewStore(REVIEW_DB_PATH)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="A valid X-API-Key is required")


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    increment("http.requests")
    response = await call_next(request)
    increment(f"http.status.{response.status_code}")
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health(_: None = Depends(require_api_key)) -> dict[str, str | bool]:
    index_ready = (VECTOR_INDEX_DIR / "policy_faiss" / "index.faiss").exists()
    return {"status": "ok", "index_ready": index_ready}


@app.get("/api/index")
def index_status(_: None = Depends(require_api_key)) -> dict[str, str | bool]:
    index_ready = (VECTOR_INDEX_DIR / "policy_faiss" / "index.faiss").exists()
    return {
        "ready": index_ready,
        "location": str(VECTOR_INDEX_DIR),
        "message": "Index is ready" if index_ready else "Run scripts/02_build_index.py first",
    }


@app.post("/api/ask", response_model=AskResponse)
def ask_policy(request: AskRequest, _: None = Depends(require_api_key)) -> AskResponse:
    question = request.question.strip()
    if not _is_question_like(question):
        raise HTTPException(
            status_code=400,
            detail="Please enter a specific HR policy question, including the relevant policy area or location.",
        )

    if not (VECTOR_INDEX_DIR / "policy_faiss" / "index.faiss").exists():
        raise HTTPException(status_code=503, detail="Vector index is not ready")

    try:
        result = ask(question, region=request.region or None, history=review_store.list_chats(limit=5))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Unable to generate policy answer: {exc}") from exc

    response = AskResponse(
        question=result.question,
        answer=result.decision.answer,
        draft_email=result.draft_email,
        confidence=result.decision.confidence,
        conflict_flag=result.decision.conflict_flag,
        next_action=result.decision.next_action,
        cited_sections=result.decision.cited_sections,
        retrieved=[item.model_dump() for item in result.retrieved],
        older_version_warning=result.older_version_warning,
        search_criteria=result.search_criteria.model_dump() if result.search_criteria else None,
    )
    review_store.add_chat(
        question=response.question,
        region=request.region or None,
        answer=response.answer,
        draft_email=response.draft_email,
        confidence=response.confidence,
        conflict_flag=response.conflict_flag,
        next_action=response.next_action,
        cited_sections=response.cited_sections,
        retrieved=response.retrieved,
        older_version_warning=response.older_version_warning,
    )
    return response


@app.get("/api/chat-history", response_model=ChatHistoryResponse)
def chat_history(limit: int = 50, _: None = Depends(require_api_key)) -> ChatHistoryResponse:
    bounded_limit = max(1, min(limit, 100))
    return ChatHistoryResponse(chats=review_store.list_chats(bounded_limit))


@app.post("/api/send-to-hr")
def send_to_hr(request: HREmailRequest, _: None = Depends(require_api_key)) -> dict[str, str]:
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
    case = review_store.create(
        question=request.question,
        region=request.region,
        answer=request.answer,
        draft_email=request.draft_email,
        citations=request.cited_sections,
    )
    return {"status": "sent", "message": "The case was sent to HR for review.", "case_id": case["case_id"]}


@app.get("/api/metrics")
def metrics(_: None = Depends(require_api_key)) -> dict[str, dict[str, int]]:
    return {"metrics": metrics_snapshot()}


@app.get("/api/reviews")
def list_reviews(status: str | None = None, _: None = Depends(require_api_key)) -> dict[str, list[dict]]:
    return {"cases": review_store.list(status)}


@app.patch("/api/reviews/{case_id}")
def update_review(case_id: str, update: ReviewUpdate, _: None = Depends(require_api_key)) -> dict:
    case = review_store.update(case_id, status=update.status, reviewer_note=update.reviewer_note)
    if case is None:
        raise HTTPException(status_code=404, detail="Review case not found")
    return case


_FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
