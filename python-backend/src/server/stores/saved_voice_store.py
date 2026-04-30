from __future__ import annotations

import time
import uuid

from server.stores.base import BaseStore


class SavedVoiceStore(BaseStore):
    def _ddl(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS saved_voices (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            description TEXT,
            audio_path  TEXT,
            model_id    TEXT,
            created_at  TEXT NOT NULL
        );
        """

    async def list_all(self) -> list[dict]:
        return await self._fetch_all(
            "SELECT * FROM saved_voices ORDER BY created_at DESC"
        )

    async def get(self, voice_id: str) -> dict | None:
        return await self._fetch_one(
            "SELECT * FROM saved_voices WHERE id = :id", {"id": voice_id}
        )

    async def delete(self, voice_id: str) -> bool:
        existing = await self.get(voice_id)
        if not existing:
            return False
        await self._execute(
            "DELETE FROM saved_voices WHERE id = :id", {"id": voice_id}
        )
        return True


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _uid() -> str:
    return uuid.uuid4().hex
