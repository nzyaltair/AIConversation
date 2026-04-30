from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_chat_threads_crud(client: AsyncClient):
    # Create thread
    resp = await client.post("/v1/chat/threads/", json={"title": "Test Thread", "model_id": "Qwen3-0.6B-GGUF"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "Test Thread"
    thread_id = data["id"]

    # List threads
    resp = await client.get("/v1/chat/threads/")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
    assert len(resp.json()) >= 1

    # Get thread
    resp = await client.get(f"/v1/chat/threads/{thread_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == thread_id

    # Update thread
    resp = await client.patch(f"/v1/chat/threads/{thread_id}", json={"title": "Updated Title"})
    assert resp.status_code == 200
    assert resp.json()["title"] == "Updated Title"

    # Send non-streaming message
    resp = await client.post(
        f"/v1/chat/threads/{thread_id}/messages",
        json={"content": "Hello world", "model": "Qwen3-0.6B-GGUF"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["role"] == "assistant"
    assert len(data["content"]) > 0

    # List messages
    resp = await client.get(f"/v1/chat/threads/{thread_id}/messages")
    assert resp.status_code == 200
    msgs = resp.json()
    assert len(msgs) == 2  # user + assistant

    # Delete thread
    resp = await client.delete(f"/v1/chat/threads/{thread_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.anyio
async def test_chat_completions_sse(client: AsyncClient):
    resp = await client.post("/v1/chat/completions", json={
        "model": "Qwen3-0.6B-GGUF",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    }, timeout=10)
    assert resp.status_code == 200
    text = resp.text
    assert "data:" in text
    assert "[DONE]" in text
    assert "chat.completion.chunk" in text
