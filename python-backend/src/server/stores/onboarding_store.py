from __future__ import annotations

import time

from server.stores.base import BaseStore


class OnboardingStore(BaseStore):
    def _ddl(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS onboarding (
            id           INTEGER PRIMARY KEY DEFAULT 1 CHECK(id = 1),
            completed    INTEGER NOT NULL DEFAULT 0,
            completed_at TEXT
        );
        INSERT OR IGNORE INTO onboarding (id, completed) VALUES (1, 0);
        """

    async def get_state(self) -> bool:
        row = await self._fetch_one("SELECT completed FROM onboarding WHERE id = 1")
        return bool(row["completed"]) if row else False

    async def complete(self) -> bool:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await self._execute(
            "UPDATE onboarding SET completed = 1, completed_at = :ts WHERE id = 1",
            {"ts": ts},
        )
        return True
