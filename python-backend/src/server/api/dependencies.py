from __future__ import annotations

import asyncio
import uuid

from fastapi import Request

from server.app_state import AppState
from server.error_handlers import server_error


def get_app_state(request: Request) -> AppState:
    return request.app.state.app_state


def get_request_id(request: Request) -> str:
    correlation_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
    request.state.correlation_id = correlation_id
    return correlation_id


async def acquire_semaphore(request: Request) -> None:
    app_state: AppState = request.app.state.app_state
    try:
        await asyncio.wait_for(
            app_state.request_semaphore.acquire(),
            timeout=app_state.config.request_timeout_secs,
        )
    except asyncio.TimeoutError:
        raise server_error("Server is busy, please try again later")
    request.state._semaphore_acquired = True
