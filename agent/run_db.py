"""Run database — SQLite-backed conversation and run tracking.

Stores conversation runs with messages, generated artifacts, and metadata.
Enables multi-turn conversations with persistent history.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    timestamp: str  # ISO format


@dataclass
class Run:
    id: str
    conversation_id: str
    user_message: str
    assistant_response: str
    created_at: str
    assets: dict  # NASA data returned
    manifest: dict  # Video generation manifest
    messages: list[Message]  # Full conversation history up to this point


class RunDB:
    """Persistent conversation and run storage."""

    def __init__(self, db_path: str | Path = "output/runs.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    title TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    assistant_response TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    assets TEXT,
                    manifest TEXT,
                    messages TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                )
            """)
            conn.commit()

    def create_conversation(self, conversation_id: str, title: Optional[str] = None) -> None:
        """Create a new conversation."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO conversations (conversation_id, created_at, title) VALUES (?, ?, ?)",
                (conversation_id, datetime.now().isoformat(), title)
            )
            conn.commit()

    def set_conversation_title(self, conversation_id: str, title: str) -> None:
        """Set the display title for a conversation (typically first user message)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE conversations SET title = ? WHERE conversation_id = ?",
                (title[:120], conversation_id),
            )
            conn.commit()

    def save_run(
        self,
        run_id: str,
        conversation_id: str,
        user_message: str,
        assistant_response: str,
        assets: dict | None = None,
        manifest: dict | None = None,
        messages: list[Message] | None = None,
    ) -> None:
        """Save a run to the database."""
        self.create_conversation(conversation_id)

        # Be defensive: messages should always be Message dataclass instances,
        # but if a caller ever hands us something else (a plain dict, a stray
        # string, etc.) we want a degraded-but-valid row, not a crashed page.
        _messages_json: list[dict] = []
        for m in messages or []:
            if is_dataclass(m) and not isinstance(m, type):
                _messages_json.append(asdict(m))
            elif isinstance(m, dict):
                _messages_json.append({
                    "role": m.get("role", "unknown"),
                    "content": m.get("content", ""),
                    "timestamp": m.get("timestamp", datetime.now().isoformat()),
                })
            else:
                _messages_json.append({
                    "role": "unknown",
                    "content": str(m),
                    "timestamp": datetime.now().isoformat(),
                })

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, conversation_id, user_message, assistant_response, created_at, assets, manifest, messages)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    conversation_id,
                    user_message,
                    assistant_response,
                    datetime.now().isoformat(),
                    json.dumps(assets or {}),
                    json.dumps(manifest or {}),
                    json.dumps(_messages_json),
                ),
            )
            conn.commit()

    def get_conversation_history(self, conversation_id: str) -> list[dict]:
        """Get all runs in a conversation, in order."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT run_id, user_message, assistant_response, created_at, assets, manifest, messages
                FROM runs
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,),
            ).fetchall()

        runs = []
        for row in rows:
            messages = json.loads(row[6]) if row[6] else []
            runs.append({
                "run_id": row[0],
                "user_message": row[1],
                "assistant_response": row[2],
                "created_at": row[3],
                "assets": json.loads(row[4]) if row[4] else {},
                "manifest": json.loads(row[5]) if row[5] else {},
                "messages": messages,
            })
        return runs

    def get_latest_run(self, conversation_id: str) -> dict | None:
        """Get the most recent run in a conversation."""
        runs = self.get_conversation_history(conversation_id)
        return runs[-1] if runs else None

    def list_conversations(self) -> list[dict]:
        """List all conversations."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT conversation_id, title, created_at FROM conversations ORDER BY created_at DESC"
            ).fetchall()
        return [
            {"conversation_id": row[0], "title": row[1] or "Untitled", "created_at": row[2]}
            for row in rows
        ]