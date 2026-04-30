from __future__ import annotations

import time
import uuid

from server.stores.base import BaseStore


class SpeechHistoryStore(BaseStore):
    def _ddl(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS speech_history (
            id                  TEXT PRIMARY KEY,
            route_kind          TEXT NOT NULL DEFAULT 'tts',
            model_id            TEXT,
            speaker             TEXT,
            input_text          TEXT,
            audio_path          TEXT,
            audio_duration_secs REAL,
            generation_time_ms  REAL,
            created_at          TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_sp_created_at
            ON speech_history(route_kind, created_at DESC);
        """

    async def list_all(self) -> list[dict]:
        return await self._fetch_all(
            "SELECT * FROM speech_history ORDER BY created_at DESC"
        )

    async def create(
        self,
        *,
        model_id: str | None = None,
        speaker: str | None = None,
        input_text: str | None = None,
        audio_path: str = "",
        audio_duration_secs: float | None = None,
        generation_time_ms: float | None = None,
        route_kind: str = "tts",
    ) -> dict:
        ts = _now()
        row_id = _uid()
        await self._execute(
            """INSERT INTO speech_history
               (id, route_kind, model_id, speaker, input_text, audio_path,
                audio_duration_secs, generation_time_ms, created_at)
               VALUES (:id,:rk,:m,:s,:t,:ap,:ad,:gt,:c)""",
            {
                "id": row_id, "rk": route_kind, "m": model_id,
                "s": speaker, "t": input_text, "ap": audio_path,
                "ad": audio_duration_secs, "gt": generation_time_ms, "c": ts,
            },
        )
        return dict(await self.get(row_id) or {})

    async def get(self, record_id: str) -> dict | None:
        return await self._fetch_one(
            "SELECT * FROM speech_history WHERE id = :id", {"id": record_id}
        )

    async def delete(self, record_id: str) -> bool:
        existing = await self.get(record_id)
        if not existing:
            return False
        await self._execute(
            "DELETE FROM speech_history WHERE id = :id", {"id": record_id}
        )
        return True


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _uid() -> str:
    return uuid.uuid4().hex
