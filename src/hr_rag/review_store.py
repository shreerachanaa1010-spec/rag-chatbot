"""SQLite-backed persistence for human HR review cases."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class ReviewStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS review_cases (
                    case_id TEXT PRIMARY KEY, question TEXT NOT NULL, region TEXT,
                    answer TEXT NOT NULL, draft_email TEXT NOT NULL, citations TEXT NOT NULL,
                    status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    reviewer_note TEXT
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS chat_history (
                    chat_id TEXT PRIMARY KEY, question TEXT NOT NULL, region TEXT,
                    answer TEXT NOT NULL, draft_email TEXT NOT NULL, confidence TEXT NOT NULL,
                    conflict_flag INTEGER NOT NULL, next_action TEXT NOT NULL,
                    cited_sections TEXT NOT NULL, retrieved TEXT NOT NULL,
                    older_version_warning TEXT,
                    created_at TEXT NOT NULL
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def create(self, *, question: str, region: str | None, answer: str, draft_email: str, citations: list[str]) -> dict:
        case_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO review_cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (case_id, question, region, answer, draft_email, json.dumps(citations), "open", now, now, None),
            )
        return self.get(case_id)  # type: ignore[return-value]

    def get(self, case_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM review_cases WHERE case_id = ?", (case_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["citations"] = json.loads(result["citations"])
        return result

    def list(self, status: str | None = None) -> list[dict]:
        query = "SELECT case_id FROM review_cases"
        parameters: tuple[str, ...] = ()
        if status:
            query += " WHERE status = ?"
            parameters = (status,)
        query += " ORDER BY created_at DESC"
        with self._connect() as connection:
            ids = [row["case_id"] for row in connection.execute(query, parameters)]
        return [self.get(case_id) for case_id in ids]  # type: ignore[list-item]

    def update(self, case_id: str, *, status: str, reviewer_note: str | None = None) -> dict | None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE review_cases SET status = ?, reviewer_note = ?, updated_at = ? WHERE case_id = ?",
                (status, reviewer_note, now, case_id),
            )
        return self.get(case_id) if cursor.rowcount else None

    def add_chat(self, *, question: str, region: str | None, answer: str, draft_email: str,
                 confidence: str, conflict_flag: bool, next_action: str,
                 cited_sections: list[str], retrieved: list[dict],
                 older_version_warning: str | None) -> dict:
        chat_id = str(uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO chat_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    chat_id, question, region, answer, draft_email, confidence,
                    int(conflict_flag), next_action, json.dumps(cited_sections),
                    json.dumps(retrieved), older_version_warning, created_at,
                ),
            )
        return self.get_chat(chat_id)  # type: ignore[return-value]

    def get_chat(self, chat_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM chat_history WHERE chat_id = ?", (chat_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["conflict_flag"] = bool(result["conflict_flag"])
        result["cited_sections"] = json.loads(result["cited_sections"])
        result["retrieved"] = json.loads(result["retrieved"])
        return result

    def list_chats(self, limit: int = 50) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT chat_id FROM chat_history ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self.get_chat(row["chat_id"]) for row in rows]  # type: ignore[list-item]