from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


WORD_RE = re.compile(r"[\wáéíóúñü]{3,}", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );
            """
        )
        self.conn.commit()

    def new_session(self) -> str:
        session_id = uuid.uuid4().hex[:12]
        now = utc_now()
        self.conn.execute(
            "INSERT INTO sessions(id, created_at, updated_at) VALUES (?, ?, ?)",
            (session_id, now, now),
        )
        self.conn.commit()
        return session_id

    def add_message(self, session_id: str, role: str, content: str) -> None:
        now = utc_now()
        self.conn.execute(
            "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now),
        )
        self.conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
        )
        self.conn.commit()

    def load_messages(self, session_id: str, limit: int = 30) -> list[dict[str, str]]:
        rows = self.conn.execute(
            """
            SELECT role, content FROM (
                SELECT id, role, content FROM messages
                WHERE session_id = ? ORDER BY id DESC LIMIT ?
            ) ORDER BY id ASC
            """,
            (session_id, limit),
        ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def remember(self, text: str) -> int:
        cursor = self.conn.execute(
            "INSERT INTO memories(text, created_at) VALUES (?, ?)", (text.strip(), utc_now())
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def forget(self, memory_id: int) -> bool:
        cursor = self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def list_memories(self, limit: int = 50) -> list[dict[str, object]]:
        rows = self.conn.execute(
            "SELECT id, text, created_at FROM memories ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]

    def relevant(self, query: str, limit: int = 6) -> list[str]:
        query_words = set(WORD_RE.findall(query.lower()))
        if not query_words:
            return []
        rows = self.conn.execute(
            "SELECT id, text FROM memories ORDER BY id DESC LIMIT 500"
        ).fetchall()
        ranked: list[tuple[float, str]] = []
        for recency, row in enumerate(rows):
            words = set(WORD_RE.findall(row["text"].lower()))
            overlap = len(query_words & words)
            if overlap:
                score = overlap / max(len(query_words), 1) + 0.03 / (recency + 1)
                ranked.append((score, row["text"]))
        ranked.sort(reverse=True)
        return [text for _, text in ranked[:limit]]

    def export_jsonl(self, target: Path) -> int:
        """Exporta pares de conversación para revisión, no entrena automáticamente."""
        rows = self.conn.execute(
            "SELECT session_id, role, content FROM messages ORDER BY session_id, id"
        ).fetchall()
        sessions: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            sessions.setdefault(row["session_id"], []).append(
                {"role": row["role"], "content": row["content"]}
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with target.open("w", encoding="utf-8") as handle:
            for messages in sessions.values():
                if any(m["role"] == "assistant" for m in messages):
                    handle.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                    count += 1
        return count

    def close(self) -> None:
        self.conn.close()

