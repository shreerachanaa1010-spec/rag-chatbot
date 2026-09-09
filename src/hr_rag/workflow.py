"""LangGraph workflow for retrieval, version handling, and HR review routing."""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from hr_rag.generation import generate_decision
from hr_rag.retrieval import current_version_chunks, older_version_warning, retrieve
from hr_rag.schemas import RetrievedChunk


class WorkflowState(TypedDict, total=False):
    question: str
    region: str | None
    retrieved: list[RetrievedChunk]
    current_chunks: list[RetrievedChunk]
    older_version_warning: str | None
    review_required: bool
    raw: dict


def retrieve_node(state: WorkflowState) -> WorkflowState:
    return {"retrieved": retrieve(state["question"], region=state.get("region"))}


def select_latest_node(state: WorkflowState) -> WorkflowState:
    retrieved = state.get("retrieved", [])
    current_chunks = current_version_chunks(retrieved)
    return {
        "current_chunks": current_chunks,
        "older_version_warning": older_version_warning(retrieved),
    }


def review_gate_node(state: WorkflowState) -> WorkflowState:
    retrieved = state.get("retrieved", [])
    current_chunks = state.get("current_chunks", [])
    doc_versions: dict[str, set[str]] = {}
    for item in retrieved:
        doc_versions.setdefault(item.chunk.doc_id, set()).add(item.chunk.version)

    unresolved_conflict = any(
        len(versions) > 1
        and not any("supersedes" in item.chunk.text.lower() for item in current_chunks)
        for versions in doc_versions.values()
    )
    return {"review_required": not current_chunks or unresolved_conflict}


def route_after_review(state: WorkflowState) -> str:
    return "hr_review" if state.get("review_required") else "generate"


def hr_review_node(state: WorkflowState) -> WorkflowState:
    return {
        "raw": {
            "answer": "The retrieved policy information requires HR review before a definitive answer can be provided.",
            "draft_email": "Dear Employee,\n\nYour question has been escalated to HR for policy review. An HR representative will follow up with you.\n\nBest regards,\nHR Support Team",
            "cited_sections": [],
            "confidence": "low",
            "conflict_flag": True,
            "next_action": "escalate_to_hr_review",
        }
    }


def generate_node(state: WorkflowState) -> WorkflowState:
    return {"raw": generate_decision(state["question"], state["current_chunks"])}


def build_workflow():
    graph = StateGraph(WorkflowState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("select_latest", select_latest_node)
    graph.add_node("review_gate", review_gate_node)
    graph.add_node("hr_review", hr_review_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "select_latest")
    graph.add_edge("select_latest", "review_gate")
    graph.add_conditional_edges(
        "review_gate",
        route_after_review,
        {"hr_review": "hr_review", "generate": "generate"},
    )
    graph.add_edge("hr_review", END)
    graph.add_edge("generate", END)
    return graph.compile()


workflow = build_workflow()
