from __future__ import annotations

from fastapi import APIRouter

from server.app_state import AppState
from server.models.schemas import UpdateVoiceProfileRequest


def create_router(state: AppState) -> APIRouter:
    router = APIRouter()
    store = state.voice_store

    @router.get("/profile")
    async def get_profile():
        profile = await store.get_or_create_profile()
        return {
            "id": profile["id"],
            "name": profile["name"],
            "system_prompt": profile["system_prompt"],
            "observational_memory_enabled": bool(profile["observational_memory_enabled"]),
            "default_system_prompt": profile["default_system_prompt"],
        }

    @router.patch("/profile")
    async def update_profile(body: UpdateVoiceProfileRequest):
        fields: dict = {}
        if body.name is not None:
            fields["name"] = body.name
        if body.system_prompt is not None:
            fields["system_prompt"] = body.system_prompt
        if body.observational_memory_enabled is not None:
            fields["observational_memory_enabled"] = body.observational_memory_enabled
        profile = await store.update_profile(**fields)
        return {
            "id": profile["id"],
            "name": profile["name"],
            "system_prompt": profile["system_prompt"],
            "observational_memory_enabled": bool(profile["observational_memory_enabled"]),
            "default_system_prompt": profile["default_system_prompt"],
        }

    return router
