from __future__ import annotations

from hr_rag.review_store import ReviewStore


def test_review_store_persists_and_updates_case(tmp_path) -> None:
    store = ReviewStore(tmp_path / "reviews.sqlite3")
    created = store.create(
        question="How much leave?",
        region="US",
        answer="HR review required.",
        draft_email="We will follow up.",
        citations=["HR-POL-US-014 v2.0"],
    )

    assert created["status"] == "open"
    updated = store.update(created["case_id"], status="resolved", reviewer_note="Confirmed with HR.")

    assert updated is not None
    assert updated["status"] == "resolved"
    assert updated["reviewer_note"] == "Confirmed with HR."


def test_review_store_persists_chat_history(tmp_path) -> None:
    store = ReviewStore(tmp_path / "reviews.sqlite3")
    created = store.add_chat(
        question="How much leave?",
        region="US",
        answer="Review the current policy.",
        draft_email="We will follow up.",
        confidence="medium",
        conflict_flag=False,
        next_action="send_to_employee",
        cited_sections=["HR-POL-US-014 v2.0"],
        retrieved=[{"chunk": {"doc_id": "HR-POL-US-014"}, "distance": 0.1}],
        older_version_warning=None,
    )

    assert created["question"] == "How much leave?"
    assert store.list_chats()[0]["retrieved"][0]["chunk"]["doc_id"] == "HR-POL-US-014"