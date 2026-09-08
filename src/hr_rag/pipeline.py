"""
Step 8: Pipeline orchestration.

Wires retrieval, conflict detection, and generation together into a single
`ask(question)` call. This is intentionally NOT agentic: it's one fixed,
linear sequence of steps that always runs the same way, which is what makes
"traditional" RAG easy to test, debug, and reason about.
"""
from __future__ import annotations

from hr_rag.generation import generate_decision
from hr_rag.retrieval import detect_version_conflicts, retrieve
from hr_rag.schemas import DecisionOutput, PipelineResult


def ask(question: str, region: str | None = None) -> PipelineResult:
    retrieved = retrieve(question, region=region)
    code_level_conflicts = detect_version_conflicts(retrieved)

    raw = generate_decision(question, retrieved)

    # Deterministic guardrail: never trust the LLM alone for conflict_flag.
    # If our own code detected 2+ policy versions in the retrieved context,
    # force escalation regardless of what the model decided.
    if code_level_conflicts:
        raw["conflict_flag"] = True
        raw["next_action"] = "escalate_to_hr_review"

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
        retrieved=retrieved,
    )
