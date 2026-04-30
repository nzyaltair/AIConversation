from __future__ import annotations

import base64
import os
import uuid

from fastapi import APIRouter

from server.app_state import AppState
from server.models.schemas import CreateSpeechHistoryRequest


def create_router(state: AppState) -> APIRouter:
    router = APIRouter()
    store = state.speech_history_store

    @router.get("/")
    async def list_all() -> list[dict]:
        rows = await store.list_all()
        return [dict(r) for r in rows]

    @router.post("/")
    async def create(body: CreateSpeechHistoryRequest):
        audio_path = ""
        if body.audio_base64:
            audio_path = os.path.join(state.config.media_dir, f"tts_{uuid.uuid4().hex}.wav")
            try:
                data = base64.b64decode(body.audio_base64)
                with open(audio_path, "wb") as f:
                    f.write(data)
            except Exception:
                audio_path = ""
        row = await store.create(
            model_id=body.model_id,
            speaker=body.speaker,
            input_text=body.input_text,
            audio_path=audio_path,
            audio_duration_secs=body.audio_duration_secs,
            generation_time_ms=body.generation_time_ms,
        )
        return dict(row)

    @router.get("/{record_id}")
    async def get_record(record_id: str):
        row = await store.get(record_id)
        if not row:
            from server.error_handlers import not_found
            raise not_found("TTS generation record not found")
        return dict(row)

    @router.delete("/{record_id}")
    async def delete_record(record_id: str):
        ok = await store.delete(record_id)
        if not ok:
            from server.error_handlers import not_found
            raise not_found("TTS generation record not found")
        return {"status": "ok"}

    return router
