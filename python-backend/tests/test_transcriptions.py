from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_transcriptions_crud(client: AsyncClient):
    # List
    resp = await client.get("/v1/transcriptions/")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    # Create (JSON form - just text, no file)
    resp = await client.post("/v1/transcriptions/", data={"text": "Test transcription"})
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_tts_history_crud(client: AsyncClient):
    resp = await client.get("/v1/text-to-speech-generations")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    resp = await client.post("/v1/text-to-speech-generations", json={
        "model_id": "Kokoro-82M",
        "input_text": "Hello world",
        "speaker": "default",
    })
    assert resp.status_code == 200
