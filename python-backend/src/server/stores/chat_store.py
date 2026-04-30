from __future__ import annotations

import time
import uuid

from server.stores.base import BaseStore


class ChatStore(BaseStore):
    def _ddl(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS chat_threads (
            id                    TEXT PRIMARY KEY,
            title                 TEXT NOT NULL DEFAULT '',
            model_id              TEXT NOT NULL DEFAULT '',
            created_at            TEXT NOT NULL,
            updated_at            TEXT NOT NULL,
            message_count         INTEGER NOT NULL DEFAULT 0,
            last_message_preview  TEXT
        );
        CREATE TABLE IF NOT EXISTS chat_messages (
            id            TEXT PRIMARY KEY,
            thread_id     TEXT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
            role          TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
            content       TEXT NOT NULL DEFAULT '',
            created_at    TEXT NOT NULL,
            model_id      TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_msg_thread_created
            ON chat_messages(thread_id, created_at, id);
        """

    # ── Threads ──
    async def list_threads(self) -> list[dict]:
        return await self._fetch_all(
            "SELECT * FROM chat_threads ORDER BY updated_at DESC"
        )

    async def create_thread(self, title: str, model_id: str = "") -> dict:
        ts = _now()
        row_id = _uid()
        await self._execute(
            "INSERT INTO chat_threads (id, title, model_id, created_at, updated_at) "
            "VALUES (:id, :t, :m, :c, :u)",
            {"id": row_id, "t": title, "m": model_id, "c": ts, "u": ts},
        )
        return await self.get_thread(row_id)

    async def get_thread(self, thread_id: str) -> dict | None:
        return await self._fetch_one(
            "SELECT * FROM chat_threads WHERE id = :id", {"id": thread_id}
        )

    async def update_thread(self, thread_id: str, **fields: str) -> dict | None:
        existing = await self.get_thread(thread_id)
        if not existing:
            return None
        sets = ["updated_at = :now"]
        params: dict[str, str] = {"id": thread_id, "now": _now()}
        for key, val in fields.items():
            if val is not None:
                sets.append(f"{key} = :{key}")
                params[key] = val
        await self._execute(
            f"UPDATE chat_threads SET {', '.join(sets)} WHERE id = :id", params
        )
        return await self.get_thread(thread_id)

    async def delete_thread(self, thread_id: str) -> bool:
        existing = await self.get_thread(thread_id)
        if not existing:
            return False
        await self._execute("DELETE FROM chat_threads WHERE id = :id", {"id": thread_id})
        return True

    # ── Messages ──
    async def list_messages(self, thread_id: str) -> list[dict]:
        return await self._fetch_all(
            "SELECT * FROM chat_messages WHERE thread_id = :tid ORDER BY created_at ASC, id ASC",
            {"tid": thread_id},
        )

    async def append_message(
        self, thread_id: str, role: str, content: str, model_id: str = ""
    ) -> dict:
        ts = _now()
        msg_id = _uid()
        await self._execute(
            "INSERT INTO chat_messages (id, thread_id, role, content, created_at, model_id) "
            "VALUES (:id, :tid, :r, :c, :ts, :m)",
            {"id": msg_id, "tid": thread_id, "r": role, "c": content, "ts": ts, "m": model_id},
        )
        # Update thread denorm columns
        new_count = await self._fetch_scalar(
            "SELECT COUNT(*) FROM chat_messages WHERE thread_id = :tid", {"tid": thread_id}
        )
        preview = content[:120] if len(content) > 120 else content
        await self._execute(
            "UPDATE chat_threads SET message_count = :c, last_message_preview = :p, "
            "updated_at = :u WHERE id = :tid",
            {"c": new_count, "p": preview, "u": ts, "tid": thread_id},
        )
        row = await self._fetch_one(
            "SELECT * FROM chat_messages WHERE id = :id", {"id": msg_id}
        )
        assert row is not None
        return row


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _uid() -> str:
    return uuid.uuid4().hex
