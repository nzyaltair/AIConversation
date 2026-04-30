from __future__ import annotations

import time
import uuid

from server.stores.base import BaseStore


class VoiceStore(BaseStore):
    def _ddl(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS voice_profiles (
            id                            TEXT PRIMARY KEY,
            name                          TEXT,
            system_prompt                 TEXT,
            observational_memory_enabled  INTEGER NOT NULL DEFAULT 0,
            default_system_prompt         TEXT NOT NULL DEFAULT
                'You are a helpful voice assistant. Be concise and direct in your responses.',
            created_at                    TEXT NOT NULL,
            updated_at                    TEXT NOT NULL
        );
        """

    _DEFAULT_PROMPT = (
        "You are a helpful voice assistant. Be concise and direct in your responses."
    )

    async def get_or_create_profile(self) -> dict:
        profile = await self._fetch_one(
            "SELECT * FROM voice_profiles ORDER BY created_at ASC LIMIT 1"
        )
        if profile:
            return profile
        ts = _now()
        row_id = "vprof_default"
        await self._execute(
            """INSERT INTO voice_profiles
               (id, name, system_prompt, observational_memory_enabled,
                default_system_prompt, created_at, updated_at)
               VALUES (:id, :n, :sp, :ome, :dsp, :c, :u)""",
            {
                "id": row_id, "n": "Assistant", "sp": self._DEFAULT_PROMPT,
                "ome": 0, "dsp": self._DEFAULT_PROMPT, "c": ts, "u": ts,
            },
        )
        profile = await self._fetch_one(
            "SELECT * FROM voice_profiles WHERE id = :id", {"id": row_id}
        )
        assert profile is not None
        return profile

    async def update_profile(self, **fields: str | bool | None) -> dict | None:
        profile = await self.get_or_create_profile()
        profile_id = profile["id"]
        sets = ["updated_at = :now"]
        params: dict[str, str] = {"id": profile_id, "now": _now()}
        for key, val in fields.items():
            if val is not None:
                if isinstance(val, bool):
                    val_str = "1" if val else "0"
                else:
                    val_str = str(val)
                sets.append(f"{key} = :{key}")
                params[key] = val_str
        await self._execute(
            f"UPDATE voice_profiles SET {', '.join(sets)} WHERE id = :id", params
        )
        return await self._fetch_one(
            "SELECT * FROM voice_profiles WHERE id = :id", {"id": profile_id}
        )


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _uid() -> str:
    return uuid.uuid4().hex
