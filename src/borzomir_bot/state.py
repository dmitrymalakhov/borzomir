from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any


class StateStore:
    def __init__(self, database_path: str, *, max_history_messages: int) -> None:
        self.database_path = database_path
        self.max_history_messages = max_history_messages
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path)
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def close(self) -> None:
        self._connection.close()

    def append_history(self, *, chat_id: int, role: str, content: str) -> None:
        if self.max_history_messages <= 0:
            return
        now = int(time.time())
        with self._connection:
            self._connection.execute(
                "INSERT INTO chat_history(chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (chat_id, role, content, now),
            )

    def list_history(self, chat_id: int, *, limit: int | None = None) -> list[dict[str, str]]:
        if limit is not None and limit <= 0:
            return []
        if limit is None:
            rows = self._connection.execute(
                "SELECT role, content FROM chat_history WHERE chat_id = ? ORDER BY id ASC",
                (chat_id,),
            ).fetchall()
        else:
            rows = self._connection.execute(
                """
                SELECT role, content FROM (
                    SELECT id, role, content
                    FROM chat_history
                    WHERE chat_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (chat_id, limit),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def list_overflow_history(self, *, chat_id: int, keep_messages: int) -> list[dict[str, Any]]:
        if keep_messages < 0:
            keep_messages = 0
        rows = self._connection.execute(
            """
            SELECT id, role, content, created_at
            FROM chat_history
            WHERE chat_id = ?
              AND id NOT IN (
                SELECT id
                FROM chat_history
                WHERE chat_id = ?
                ORDER BY id DESC
                LIMIT ?
              )
            ORDER BY id ASC
            """,
            (chat_id, chat_id, keep_messages),
        ).fetchall()
        return [
            {"id": row["id"], "role": row["role"], "content": row["content"], "created_at": row["created_at"]}
            for row in rows
        ]

    def delete_history_ids(self, ids: list[int]) -> int:
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self._connection:
            cursor = self._connection.execute(f"DELETE FROM chat_history WHERE id IN ({placeholders})", ids)
        return cursor.rowcount

    def reset_history(self, chat_id: int) -> None:
        with self._connection:
            self._connection.execute("DELETE FROM chat_history WHERE chat_id = ?", (chat_id,))
            self._connection.execute("DELETE FROM chat_summaries WHERE chat_id = ?", (chat_id,))

    def history_count(self, chat_id: int) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) AS count FROM chat_history WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()
        return int(row["count"] or 0)

    def history_stats(self) -> dict[str, int]:
        row = self._connection.execute(
            "SELECT COUNT(DISTINCT chat_id) AS chats, COUNT(*) AS messages FROM chat_history"
        ).fetchone()
        return {"chats": int(row["chats"] or 0), "messages": int(row["messages"] or 0)}

    def get_summary(self, chat_id: int) -> str | None:
        row = self._connection.execute(
            "SELECT content FROM chat_summaries WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()
        if row is None:
            return None
        return str(row["content"])

    def set_summary(self, *, chat_id: int, content: str | None) -> None:
        with self._connection:
            if content is None or not content.strip():
                self._connection.execute("DELETE FROM chat_summaries WHERE chat_id = ?", (chat_id,))
                return
            self._connection.execute(
                """
                INSERT INTO chat_summaries(chat_id, content, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    content = excluded.content,
                    updated_at = excluded.updated_at
                """,
                (chat_id, content.strip(), int(time.time())),
            )

    def add_note(self, *, chat_id: int, content: str) -> None:
        now = int(time.time())
        with self._connection:
            self._connection.execute(
                "INSERT INTO memory_notes(chat_id, content, created_at) VALUES (?, ?, ?)",
                (chat_id, content, now),
            )

    def list_notes(self, chat_id: int) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT id, content, created_at FROM memory_notes WHERE chat_id = ? ORDER BY id ASC",
            (chat_id,),
        ).fetchall()
        return [{"id": row["id"], "content": row["content"], "created_at": row["created_at"]} for row in rows]

    def clear_notes(self, chat_id: int) -> int:
        with self._connection:
            cursor = self._connection.execute("DELETE FROM memory_notes WHERE chat_id = ?", (chat_id,))
        return cursor.rowcount

    def get_chat_settings(self, chat_id: int) -> dict[str, Any]:
        row = self._connection.execute(
            "SELECT system_prompt, temperature, max_tokens, model FROM chat_settings WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()
        if row is None:
            return {}
        return {
            "system_prompt": row["system_prompt"],
            "temperature": row["temperature"],
            "max_tokens": row["max_tokens"],
            "model": row["model"],
        }

    def set_system_prompt(self, *, chat_id: int, system_prompt: str | None) -> None:
        self._upsert_chat_setting(chat_id=chat_id, key="system_prompt", value=system_prompt)

    def set_temperature(self, *, chat_id: int, temperature: float | None) -> None:
        self._upsert_chat_setting(chat_id=chat_id, key="temperature", value=temperature)

    def set_max_tokens(self, *, chat_id: int, max_tokens: int | None) -> None:
        self._upsert_chat_setting(chat_id=chat_id, key="max_tokens", value=max_tokens)

    def set_model(self, *, chat_id: int, model: str | None) -> None:
        self._upsert_chat_setting(chat_id=chat_id, key="model", value=model)

    def add_allowed_user(self, user_id: int) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO allowed_users(user_id, created_at) VALUES (?, ?)",
                (user_id, int(time.time())),
            )

    def remove_allowed_user(self, user_id: int) -> bool:
        with self._connection:
            cursor = self._connection.execute("DELETE FROM allowed_users WHERE user_id = ?", (user_id,))
        return cursor.rowcount > 0

    def clear_allowed_users(self) -> int:
        with self._connection:
            cursor = self._connection.execute("DELETE FROM allowed_users")
        return cursor.rowcount

    def allowed_user_ids(self) -> frozenset[int]:
        rows = self._connection.execute("SELECT user_id FROM allowed_users ORDER BY user_id ASC").fetchall()
        return frozenset(int(row["user_id"]) for row in rows)

    def stats(self) -> dict[str, int]:
        history = self.history_stats()
        notes_row = self._connection.execute("SELECT COUNT(*) AS count FROM memory_notes").fetchone()
        summaries_row = self._connection.execute("SELECT COUNT(*) AS count FROM chat_summaries").fetchone()
        summary_chars_row = self._connection.execute(
            "SELECT COALESCE(SUM(LENGTH(content)), 0) AS count FROM chat_summaries"
        ).fetchone()
        allowed_row = self._connection.execute("SELECT COUNT(*) AS count FROM allowed_users").fetchone()
        return {
            **history,
            "notes": int(notes_row["count"] or 0),
            "summaries": int(summaries_row["count"] or 0),
            "summary_chars": int(summary_chars_row["count"] or 0),
            "allowed_users": int(allowed_row["count"] or 0),
        }

    def _initialize(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chat_history_chat_id_id
                    ON chat_history(chat_id, id);

                CREATE TABLE IF NOT EXISTS memory_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memory_notes_chat_id_id
                    ON memory_notes(chat_id, id);

                CREATE TABLE IF NOT EXISTS chat_settings (
                    chat_id INTEGER PRIMARY KEY,
                    system_prompt TEXT,
                    temperature REAL,
                    max_tokens INTEGER,
                    model TEXT
                );

                CREATE TABLE IF NOT EXISTS allowed_users (
                    user_id INTEGER PRIMARY KEY,
                    created_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_summaries (
                    chat_id INTEGER PRIMARY KEY,
                    content TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                """
            )
            columns = {
                row["name"]
                for row in self._connection.execute("PRAGMA table_info(chat_settings)").fetchall()
            }
            if "model" not in columns:
                self._connection.execute("ALTER TABLE chat_settings ADD COLUMN model TEXT")

    def _upsert_chat_setting(self, *, chat_id: int, key: str, value: Any) -> None:
        if key not in {"system_prompt", "temperature", "max_tokens", "model"}:
            raise ValueError(f"Unsupported chat setting: {key}")
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO chat_settings(chat_id, system_prompt, temperature, max_tokens, model)
                VALUES (?, NULL, NULL, NULL, NULL)
                ON CONFLICT(chat_id) DO NOTHING
                """,
                (chat_id,),
            )
            self._connection.execute(f"UPDATE chat_settings SET {key} = ? WHERE chat_id = ?", (value, chat_id))
