from __future__ import annotations

from typing import Any
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy import event, text


class BaseStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._engine: AsyncEngine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}", echo=False
        )
        self._register_pragmas()

    def _register_pragmas(self) -> None:
        @event.listens_for(self._engine.sync_engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection: Any, connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=3000")
            cursor.close()

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    async def initialize(self) -> None:
        async with self._engine.begin() as conn:
            for stmt in self._ddl().split(";"):
                stmt = stmt.strip()
                if stmt:
                    await conn.execute(text(stmt))

    async def dispose(self) -> None:
        await self._engine.dispose()

    def _ddl(self) -> str:
        raise NotImplementedError

    async def _execute(self, sql: str, params: dict[str, Any] | None = None) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(text(sql), params or {})

    async def _execute_returning(self, sql: str, params: dict[str, Any] | None = None) -> Any:
        async with self._engine.begin() as conn:
            result = await conn.execute(text(sql), params or {})
            return result.fetchone()

    async def _fetch_all(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        async with self._engine.connect() as conn:
            result = await conn.execute(text(sql), params or {})
            rows = result.fetchall()
            return [dict(row._mapping) for row in rows]

    async def _fetch_one(self, sql: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        async with self._engine.connect() as conn:
            result = await conn.execute(text(sql), params or {})
            row = result.fetchone()
            return dict(row._mapping) if row is not None else None

    async def _fetch_scalar(self, sql: str, params: dict[str, Any] | None = None) -> Any:
        async with self._engine.connect() as conn:
            result = await conn.execute(text(sql), params or {})
            row = result.fetchone()
            return row[0] if row is not None else None
