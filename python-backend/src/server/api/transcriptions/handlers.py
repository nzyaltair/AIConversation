from __future__ import annotations

import os
import time
import json
import asyncio
import uuid

from fastapi import APIRouter, UploadFile, File, Form, Request
from fastapi.responses import FileResponse

from server.app_state import AppState


def create_router(state: AppState) -> APIRouter:
    router = APIRouter()
    store = state.transcription_store

    @router.get("/")
    async def list_all() -> list[dict]:
        rows = await store.list_all()
        return [dict(r) for r in rows]

    @router.post("/")
    async def create(
        file: UploadFile = File(None),
        text: str = Form(""),
    ):
        file_name = ""
        if file and file.filename:
            file_name = file.filename
            audio_path = os.path.join(state.config.media_dir, f"tx_{uuid.uuid4().hex}.wav")
            content = await file.read()
            with open(audio_path, "wb") as f:
                f.write(content)
        else:
            audio_path = ""
        row = await store.create(file_name=file_name, audio_path=audio_path, text=text or "Transcription placeholder")
        return dict(row)

    @router.get("/{record_id}")
    async def get_record(record_id: str):
        row = await store.get(record_id)
        if not row:
            from server.error_handlers import not_found
            raise not_found("Transcription record not found")
        return dict(row)

    @router.delete("/{record_id}")
    async def delete_record(record_id: str):
        ok = await store.delete(record_id)
        if not ok:
            from server.error_handlers import not_found
            raise not_found("Transcription record not found")
        return {"status": "ok"}

    @router.get("/{record_id}/audio")
    async def get_audio(record_id: str):
        row = await store.get(record_id)
        if not row or not row.get("audio_path"):
            from server.error_handlers import not_found
            raise not_found("Audio not found")
        path = row["audio_path"]
        if not os.path.exists(path):
            from server.error_handlers import not_found
            raise not_found("Audio file missing")
        return FileResponse(path, media_type="audio/wav")

    return router
