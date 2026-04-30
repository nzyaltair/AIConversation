from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_audio_transcriptions(client: AsyncClient):
    import io
    # Create a small "audio" file
    fake_wav = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\xbb\x00\x00\x00\xee\x02\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    files = {"file": ("test.wav", io.BytesIO(fake_wav), "audio/wav")}
    resp = await client.post("/v1/audio/transcriptions", files=files, data={"model": "whisper-base"})
    assert resp.status_code == 200
    data = resp.json()
    assert "text" in data


@pytest.mark.anyio
async def test_audio_speech(client: AsyncClient):
    resp = await client.post("/v1/audio/speech", json={
        "model": "Kokoro-82M",
        "input": "Hello world",
        "stream": False,
    })
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_audio_speech_stream(client: AsyncClient):
    resp = await client.post("/v1/audio/speech", json={
        "model": "Kokoro-82M",
        "input": "Hello world",
        "stream": True,
    }, timeout=10)
    assert resp.status_code == 200
    assert "event" in resp.text
    assert "chunk" in resp.text or "done" in resp.text
