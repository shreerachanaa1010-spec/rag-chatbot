from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app


def test_health_reports_faiss_index_ready() -> None:
    response = TestClient(app).get("/api/health")

    assert response.status_code == 200
    assert response.json()["index_ready"] is True


def test_rejects_non_question_input() -> None:
    response = TestClient(app).post("/api/ask", json={"question": "sakmfklhsdgedds"})

    assert response.status_code == 400
    assert "specific HR policy question" in response.json()["detail"]


def test_send_to_hr_uses_email_service(monkeypatch) -> None:
    sent = {}

    def fake_send(subject: str, body: str) -> None:
        sent["subject"] = subject
        sent["body"] = body

    monkeypatch.setattr("backend.main.send_hr_email", fake_send)
    response = TestClient(app).post(
        "/api/send-to-hr",
        json={
            "question": "How does parental leave work in the US?",
            "answer": "HR review is required.",
            "draft_email": "Dear Employee, HR will follow up.",
            "cited_sections": ["HR-POL-US-014 v2.0 - How to Apply"],
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "sent"
    assert "parental leave" in sent["subject"]
    assert "HR-POL-US-014" in sent["body"]
