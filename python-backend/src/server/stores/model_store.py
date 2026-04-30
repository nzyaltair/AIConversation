from __future__ import annotations

import time

from sqlalchemy import text

from server.stores.base import BaseStore


class ModelStore(BaseStore):
    _MIGRATIONS = [
        ("runtime", "ALTER TABLE model_catalog ADD COLUMN runtime TEXT NOT NULL DEFAULT ''"),
    ]

    def _ddl(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS model_catalog (
            variant          TEXT PRIMARY KEY,
            repo_id          TEXT NOT NULL,
            category         TEXT NOT NULL CHECK(category IN ('asr','llm','tts','vad')),
            runtime          TEXT NOT NULL DEFAULT '',
            status           TEXT NOT NULL DEFAULT 'not_downloaded'
                              CHECK(status IN ('not_downloaded','downloading','downloaded','ready','error')),
            enabled          INTEGER NOT NULL DEFAULT 1,
            size_bytes       INTEGER,
            downloaded_bytes INTEGER NOT NULL DEFAULT 0,
            error_message    TEXT,
            storage_path     TEXT,
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL
        );
        """

    async def initialize(self) -> None:
        await super().initialize()
        existing_cols: set[str] = set()
        async with self.engine.connect() as conn:
            result = await conn.execute(text("PRAGMA table_info(model_catalog)"))
            for row in result.fetchall():
                existing_cols.add(row[1])
        for col_name, alter_sql in self._MIGRATIONS:
            if col_name not in existing_cols:
                await self._execute(alter_sql)

    async def list_models(
        self, category: str | None = None, sort_by: str | None = None
    ) -> list[dict]:
        sql = "SELECT * FROM model_catalog"
        params: dict = {}
        if category:
            sql += " WHERE category = :category"
            params["category"] = category
        if sort_by == "size_asc":
            sql += " ORDER BY size_bytes ASC"
        elif sort_by == "size_desc":
            sql += " ORDER BY size_bytes DESC"
        else:
            sql += " ORDER BY size_bytes ASC"
        return await self._fetch_all(sql, params)

    async def get_model(self, variant: str) -> dict | None:
        return await self._fetch_one(
            "SELECT * FROM model_catalog WHERE variant = :v",
            {"v": variant},
        )

    async def upsert_model(
        self,
        variant: str,
        repo_id: str,
        category: str,
        runtime: str = "",
        size_bytes: int | None = None,
        status: str = "not_downloaded",
        enabled: bool = True,
        storage_path: str | None = None,
        downloaded_bytes: int = 0,
        error_message: str | None = None,
    ) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        existing = await self.get_model(variant)
        if existing:
            await self._execute(
                """UPDATE model_catalog
                   SET repo_id = :repo_id, category = :category,
                       runtime = :runtime,
                       size_bytes = :size_bytes, status = :status,
                       enabled = :enabled, storage_path = :storage_path,
                       downloaded_bytes = :downloaded_bytes,
                       error_message = :error_message, updated_at = :ts
                   WHERE variant = :variant""",
                {
                    "variant": variant, "repo_id": repo_id, "category": category,
                    "runtime": runtime,
                    "size_bytes": size_bytes, "status": status,
                    "enabled": int(enabled), "storage_path": storage_path,
                    "downloaded_bytes": downloaded_bytes,
                    "error_message": error_message, "ts": ts,
                },
            )
        else:
            await self._execute(
                """INSERT INTO model_catalog
                   (variant, repo_id, category, runtime, size_bytes, status, enabled,
                    storage_path, downloaded_bytes, error_message, created_at, updated_at)
                   VALUES (:variant, :repo_id, :category, :runtime, :size_bytes, :status, :enabled,
                           :storage_path, :downloaded_bytes, :error_message, :ts, :ts)""",
                {
                    "variant": variant, "repo_id": repo_id, "category": category,
                    "runtime": runtime,
                    "size_bytes": size_bytes, "status": status,
                    "enabled": int(enabled), "storage_path": storage_path,
                    "downloaded_bytes": downloaded_bytes,
                    "error_message": error_message, "ts": ts,
                },
            )

    async def update_status(
        self,
        variant: str,
        status: str,
        downloaded_bytes: int | None = None,
        error_message: str | None = None,
        storage_path: str | None = None,
    ) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        parts = ["status = :status", "updated_at = :ts"]
        params: dict = {"variant": variant, "status": status, "ts": ts}
        if downloaded_bytes is not None:
            parts.append("downloaded_bytes = :downloaded_bytes")
            params["downloaded_bytes"] = downloaded_bytes
        if error_message is not None:
            parts.append("error_message = :error_message")
            params["error_message"] = error_message
        if storage_path is not None:
            parts.append("storage_path = :storage_path")
            params["storage_path"] = storage_path
        await self._execute(
            f"UPDATE model_catalog SET {', '.join(parts)} WHERE variant = :variant",
            params,
        )

    async def delete_model(self, variant: str) -> None:
        await self._execute(
            "DELETE FROM model_catalog WHERE variant = :v", {"v": variant}
        )
