from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_voice_profile(client: AsyncClient):
    # Get profile (auto-creates)
    resp = await client.get("/v1/voice/profile")
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    assert "default_system_prompt" in data

    # Update profile
    resp = await client.patch("/v1/voice/profile", json={
        "name": "Test Voice",
        "system_prompt": "Be helpful",
        "observational_memory_enabled": True,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Test Voice"
    assert data["observational_memory_enabled"] is True


@pytest.mark.anyio
async def test_voice_observations(client: AsyncClient):
    # Add observation
    resp = await client.post("/v1/voice/observations", json={
        "summary": "User prefers short answers",
        "category": "preference",
        "confidence": 0.9,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["summary"] == "User prefers short answers"
    obs_id = data["id"]

    # List
    resp = await client.get("/v1/voice/observations")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    # Delete one
    resp = await client.delete(f"/v1/voice/observations/{obs_id}")
    assert resp.status_code == 200

    # Clear all
    resp = await client.delete("/v1/voice/observations")
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_saved_voices(client: AsyncClient):
    resp = await client.get("/v1/voices")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.anyio
async def test_onboarding(client: AsyncClient):
    resp = await client.get("/v1/onboarding/")
    assert resp.status_code == 200
    assert "completed" in resp.json()

    resp = await client.post("/v1/onboarding/complete")
    assert resp.status_code == 200
    assert resp.json()["completed"] is True


@pytest.mark.anyio
async def test_agent_sessions(client: AsyncClient):
    resp = await client.post("/v1/agent/sessions", json={"agent_id": "voice-agent"})
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    session_id = data["id"]

    resp = await client.get(f"/v1/agent/sessions/{session_id}")
    assert resp.status_code == 200

    resp = await client.post(f"/v1/agent/sessions/{session_id}/turns", json={
        "input": "Hello agent",
    })
    assert resp.status_code == 200
    assert "assistant_text" in resp.json()
