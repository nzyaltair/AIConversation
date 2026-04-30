from __future__ import annotations

from fastapi import APIRouter

from server.app_state import AppState
from server.models.schemas import AddVoiceObservationRequest


def create_router(state: AppState) -> APIRouter:
    router = APIRouter()

    async def _profile_id():
        profile = await state.voice_store.get_or_create_profile()
        return profile["id"]

    @router.get("/observations")
    async def list_observations() -> list[dict]:
        pid = await _profile_id()
        rows = await state.voice_observation_store.list_all(pid)
        return [dict(r) for r in rows]

    @router.post("/observations")
    async def add_observation(body: AddVoiceObservationRequest):
        pid = await _profile_id()
        row = await state.voice_observation_store.add_observation(
            profile_id=pid,
            category=body.category or "general",
            summary=body.summary,
            confidence=body.confidence or 0.0,
            source_text=body.source_text,
        )
        return dict(row)

    @router.delete("/observations/{observation_id}")
    async def delete_observation(observation_id: str):
        ok = await state.voice_observation_store.delete(observation_id)
        if not ok:
            from server.error_handlers import not_found
            raise not_found("Observation not found")
        return {"status": "ok"}

    @router.delete("/observations")
    async def clear_observations():
        pid = await _profile_id()
        count = await state.voice_observation_store.clear_all(pid)
        return {"status": f"{count} observations cleared"}

    return router
