from __future__ import annotations

import time
import uuid

from fastapi import APIRouter

from server.app_state import AppState
from server.models.schemas import (
    AgentSessionRequest,
    AgentTurnRequest,
    AgentTurnResponse,
)


def create_router(state: AppState) -> APIRouter:
    router = APIRouter()

    # In-memory agent session store (matches frontend's expected behavior)
    if not hasattr(state, "_agent_sessions"):
        state._agent_sessions: dict[str, dict] = {}

    @router.post("/sessions")
    async def create_session(body: AgentSessionRequest) -> dict:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        session_id = f"agent_sess_{uuid.uuid4().hex[:12]}"
        thread_id = f"thread_{uuid.uuid4().hex[:12]}"
        session = {
            "id": session_id,
            "agent_id": body.agent_id or "voice-agent",
            "thread_id": thread_id,
            "model_id": body.model_id or "Qwen3-1.7B-GGUF",
            "planning_mode": body.planning_mode or "auto",
            "created_at": ts,
            "updated_at": ts,
        }
        state._agent_sessions[session_id] = session
        return dict(session)

    @router.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        session = state._agent_sessions.get(session_id)
        if not session:
            from server.error_handlers import not_found
            raise not_found("Agent session not found")
        return dict(session)

    @router.post("/sessions/{session_id}/turns")
    async def create_turn(session_id: str, body: AgentTurnRequest) -> dict:
        session = state._agent_sessions.get(session_id)
        if not session:
            from server.error_handlers import not_found
            raise not_found("Agent session not found")
        return {
            "session_id": session_id,
            "thread_id": session.get("thread_id"),
            "model_id": body.model_id or session.get("model_id"),
            "assistant_text": f"Simulated agent response to: {body.input[:200]}",
            "plan": None,
            "tool_calls": [],
            "events": [],
        }

    return router
