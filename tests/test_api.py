from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app


def test_health_reports_faiss_index_ready() -> None:
    response = TestClient(app).get("/api/health")

    assert response.status_code == 200
    assert response.json()["index_ready"] is True
