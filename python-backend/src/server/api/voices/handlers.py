from __future__ import annotations

from fastapi import APIRouter

from server.app_state import AppState


def create_router(state: AppState) -> APIRouter:
    router = APIRouter()
    store = state.saved_voice_store

    @router.get("/")
    async def list_all() -> list[dict]:
        rows = await store.list_all()
        return [dict(r) for r in rows]

    @router.delete("/{voice_id}")
    async def delete_voice(voice_id: str):
        ok = await store.delete(voice_id)
        if not ok:
            from server.error_handlers import not_found
            raise not_found("Saved voice not found")
        return {"status": "ok"}

    return router
