from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_list_models(client: AsyncClient):
    resp = await client.get("/v1/admin/models/")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "variant" in data[0]
    assert "status" in data[0]


@pytest.mark.anyio
async def test_get_model(client: AsyncClient):
    resp = await client.get("/v1/admin/models/Qwen3-ASR-0.6B-gguf")
    assert resp.status_code == 200
    data = resp.json()
    assert data["variant"] == "Qwen3-ASR-0.6B-gguf"


@pytest.mark.anyio
async def test_download_cancel_delete(client: AsyncClient):
    variant = "Kokoro-82M-v1.1-zh-ONNX-q4"
    # Download
    resp = await client.post(f"/v1/admin/models/{variant}/download")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    # Cancel
    resp = await client.post(f"/v1/admin/models/{variant}/download/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    # Delete
    resp = await client.delete(f"/v1/admin/models/{variant}")
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_scan_disk(client: AsyncClient):
    resp = await client.get("/v1/admin/models/scan-disk")
    assert resp.status_code == 200
    data = resp.json()
    assert "disk_models" in data
    assert "orphaned" in data
    assert "catalog_count" in data
    assert isinstance(data["disk_models"], list)
    assert isinstance(data["orphaned"], list)


@pytest.mark.anyio
async def test_download_progress_sse(client: AsyncClient):
    resp = await client.get(
        "/v1/admin/models/Qwen3-ASR-0.6B-gguf/download/progress", timeout=10
    )
    assert resp.status_code == 200
