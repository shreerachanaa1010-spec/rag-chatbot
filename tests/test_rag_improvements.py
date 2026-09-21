from __future__ import annotations

from hr_rag.evaluation import citation_accuracy, groundedness
from hr_rag.query_analysis import analyze_query
from hr_rag.schemas import Chunk, RetrievedChunk
from hr_rag.validation import validate_citations


def _evidence() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=Chunk(
                chunk_id="leave_v2_000",
                doc_id="HR-POL-US-014",
                version="v2.0",
                region="US",
                effective_date="2023-03-01",
                title="Parental Leave",
                source_file="leave.md",
                chunk_index=0,
                text="Full-time employees receive 12 weeks of fully paid parental leave.",
                policy_area="parental_leave",
            ),
            distance=0.1,
        )
    ]


def test_query_analysis_extracts_search_criteria() -> None:
    criteria = analyze_query("How many weeks of parental leave do full-time employees get in the US?")

    assert criteria.policy_area == "parental_leave"
    assert criteria.region == "US"
    assert criteria.employee_type == "full_time"
    assert criteria.intent == "amount"


def test_short_follow_up_uses_recent_question_context() -> None:
    criteria = analyze_query("What about the deadline?", history=[{"question": "How much parental leave do I get in the US?"}])

    assert "parental leave" in criteria.rewritten_query.lower()
    assert criteria.policy_area == "parental_leave"


def test_invalid_citation_forces_escalation() -> None:
    result = validate_citations(
        {"answer": "Answer", "cited_sections": ["FAKE v9.0"], "draft_email": "Email"},
        _evidence(),
    )

    assert result["next_action"] == "escalate_to_hr_review"
    assert result["confidence"] == "low"


def test_advanced_evaluation_scores_grounded_answer() -> None:
    evidence = _evidence()
    answer = "Full-time employees receive 12 weeks of fully paid parental leave."

    assert citation_accuracy(["HR-POL-US-014 v2.0 - Leave Entitlement"], evidence) == 1.0
    assert groundedness(answer, evidence) > 0.5