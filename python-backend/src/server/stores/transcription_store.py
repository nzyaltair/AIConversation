from __future__ import annotations

import time
import uuid

from server.stores.base import BaseStore


class TranscriptionStore(BaseStore):
    def _ddl(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS transcription_records (
            id              TEXT PRIMARY KEY,
            file_name       TEXT NOT NULL DEFAULT '',
            audio_path      TEXT,
            duration_secs   REAL,
            language        TEXT,
            model_id        TEXT,
            text            TEXT,
            segments_json   TEXT,
            words_json      TEXT,
            status          TEXT NOT NULL DEFAULT 'completed',
            created_at      TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tx_created_at
            ON transcription_records(created_at DESC);
        """

    async def list_all(self) -> list[dict]:
        return await self._fetch_all(
            "SELECT * FROM transcription_records ORDER BY created_at DESC"
        )

    async def create(
        self,
        file_name: str = "",
        audio_path: str = "",
        duration_secs: float | None = None,
        language: str | None = None,
        model_id: str | None = None,
        text: str | None = None,
        segments_json: str = "[]",
        words_json: str = "[]",
    ) -> dict:
        ts = _now()
        row_id = _uid()
        await self._execute(
            """INSERT INTO transcription_records
               (id, file_name, audio_path, duration_secs, language, model_id, text,
                segments_json, words_json, status, created_at)
               VALUES (:id,:fn,:ap,:d,:l,:m,:t,:s,:w,:st,:c)""",
            {
                "id": row_id, "fn": file_name, "ap": audio_path,
                "d": duration_secs, "l": language, "m": model_id,
                "t": text, "s": segments_json, "w": words_json,
                "st": "completed", "c": ts,
            },
        )
        return dict(await self.get(row_id) or {})

    async def get(self, record_id: str) -> dict | None:
        return await self._fetch_one(
            "SELECT * FROM transcription_records WHERE id = :id", {"id": record_id}
        )

    async def delete(self, record_id: str) -> bool:
        existing = await self.get(record_id)
        if not existing:
            return False
        await self._execute(
            "DELETE FROM transcription_records WHERE id = :id", {"id": record_id}
        )
        return True


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _uid() -> str:
    return uuid.uuid4().hex
