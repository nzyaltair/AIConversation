from __future__ import annotations

from fastapi import APIRouter

from server.app_state import AppState


def create_router(state: AppState) -> APIRouter:
    router = APIRouter()
    store = state.onboarding_store

    @router.get("/")
    async def get_state() -> dict:
        completed = await store.get_state()
        return {"completed": completed}

    @router.post("/complete")
    async def complete() -> dict:
        await store.complete()
        return {"completed": True}

    return router
