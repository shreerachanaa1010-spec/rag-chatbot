"""Application entry point for the LangGraph HR policy workflow."""
from __future__ import annotations

from hr_rag.schemas import DecisionOutput, PipelineResult
from hr_rag.workflow import workflow


def ask(question: str, region: str | None = None) -> PipelineResult:
    state = workflow.invoke({"question": question, "region": region})
    raw = state["raw"]

    decision = DecisionOutput(
        answer=raw["answer"],
        cited_sections=raw.get("cited_sections", []),
        confidence=raw.get("confidence", "low"),
        conflict_flag=raw.get("conflict_flag", False),
        next_action=raw.get("next_action", "escalate_to_hr_review"),
    )

    return PipelineResult(
        question=question,
        decision=decision,
        draft_email=raw.get("draft_email", ""),
        retrieved=state.get("retrieved", []),
        older_version_warning=state.get("older_version_warning"),
    )
