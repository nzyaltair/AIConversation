from __future__ import annotations

import pytest
import pytest_asyncio
import tempfile
import os
import sys

# Ensure the server package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from httpx import AsyncClient, ASGITransport

from server.config import ServeConfig
from server.app_state import AppState
from server.main import create_app


@pytest_asyncio.fixture
async def tmp_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.sqlite3")
        media_dir = os.path.join(tmpdir, "media")
        os.makedirs(media_dir, exist_ok=True)
        yield ServeConfig(db_path=db_path, media_dir=media_dir)


@pytest_asyncio.fixture
async def app_state(tmp_config):
    state = AppState(tmp_config)
    await state.initialize()
    yield state
    await state.shutdown()


@pytest_asyncio.fixture
async def app(tmp_config):
    """Create a fresh FastAPI app with test config."""
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from server.api.router import build_router
    from server.error_handlers import ApiError, api_error_handler, general_exception_handler

    state = AppState(tmp_config)
    await state.initialize()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.app_state = state
        yield
        await state.shutdown()

    app = FastAPI(lifespan=lifespan)
    app.state.app_state = state
    app.add_exception_handler(ApiError, api_error_handler)
    app.add_exception_handler(Exception, general_exception_handler)
    build_router(app, state)
    return app


@pytest_asyncio.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
