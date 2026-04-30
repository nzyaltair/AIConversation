from __future__ import annotations

import time
import uuid

from server.stores.base import BaseStore


class VoiceObservationStore(BaseStore):
    def _ddl(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS voice_observations (
            id              TEXT PRIMARY KEY,
            profile_id      TEXT NOT NULL REFERENCES voice_profiles(id) ON DELETE CASCADE,
            category        TEXT NOT NULL DEFAULT 'general',
            summary         TEXT NOT NULL DEFAULT '',
            confidence      REAL NOT NULL DEFAULT 0.0,
            source_text     TEXT,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_vo_profile_created
            ON voice_observations(profile_id, created_at DESC);
        """

    async def list_all(self, profile_id: str) -> list[dict]:
        return await self._fetch_all(
            "SELECT * FROM voice_observations WHERE profile_id = :pid ORDER BY created_at DESC",
            {"pid": profile_id},
        )

    async def add_observation(
        self,
        profile_id: str,
        category: str = "general",
        summary: str = "",
        confidence: float = 0.0,
        source_text: str | None = None,
    ) -> dict:
        ts = _now()
        row_id = _uid()
        await self._execute(
            """INSERT INTO voice_observations
               (id, profile_id, category, summary, confidence, source_text,
                created_at, updated_at)
               VALUES (:id,:pid,:cat,:sum,:conf,:src,:c,:u)""",
            {
                "id": row_id, "pid": profile_id, "cat": category,
                "sum": summary, "conf": confidence, "src": source_text,
                "c": ts, "u": ts,
            },
        )
        row = await self._fetch_one(
            "SELECT * FROM voice_observations WHERE id = :id", {"id": row_id}
        )
        assert row is not None
        return row

    async def get(self, observation_id: str) -> dict | None:
        return await self._fetch_one(
            "SELECT * FROM voice_observations WHERE id = :id", {"id": observation_id}
        )

    async def delete(self, observation_id: str) -> bool:
        existing = await self.get(observation_id)
        if not existing:
            return False
        await self._execute(
            "DELETE FROM voice_observations WHERE id = :id", {"id": observation_id}
        )
        return True

    async def clear_all(self, profile_id: str) -> int:
        result = await self._fetch_scalar(
            "SELECT COUNT(*) FROM voice_observations WHERE profile_id = :pid",
            {"pid": profile_id},
        )
        count = result or 0
        await self._execute(
            "DELETE FROM voice_observations WHERE profile_id = :pid", {"pid": profile_id}
        )
        return count


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _uid() -> str:
    return uuid.uuid4().hex
